"""Controlled DLite context/head experiments; source memory never contains draft guesses.

Extra evidence has identity RoPE (non-temporal memory); pivot=0 and local
queries=1..B in EVERY experiment. These controls are not native DFlash replicas.
"""

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

STATIC_EXPERIMENTS = {
    "pivot_parallel": ("pivot", "parallel"),
    "pivot_rnn": ("pivot", "rnn"),
    "pivot_visual_parallel": ("pivot_visual", "parallel"),
    "pivot_visual_rnn": ("pivot_visual", "rnn"),
    "pivot_history_parallel": ("pivot_history", "parallel"),
    "pivot_history_rnn": ("pivot_history", "rnn"),
}
EXPERIMENTS = {
    **STATIC_EXPERIMENTS,
    "visual_static_slots": ("visual_static_slots", "rnn"),
    "visual_pivot_conditioned": ("visual_pivot_conditioned", "rnn"),
    "visual_recurrent": ("visual_recurrent", "rnn"),
}


def experiment_settings(name):
    if name not in EXPERIMENTS:
        raise ValueError(
            f"unknown visual-KV experiment {name!r}; choose {list(EXPERIMENTS)}"
        )
    mode, head = EXPERIMENTS[name]
    return dict(
        experiment_name=name,
        context_mode=mode,
        prediction_head=head,
        evidence_rope="identity",
        local_query_origin=1,
    )


class ParallelHead(nn.Module):
    """Independent low-rank predictions; previous token IDs are never consumed."""

    def __init__(self, hidden_size, rank, vocab_size):
        super().__init__()
        self.hidden_proj = nn.Linear(hidden_size, rank)
        self.output_proj = nn.Linear(rank, vocab_size, bias=False)

    def forward_teacher_forcing(self, *, hidden_states, prev_token_ids):
        return self.hidden_proj(hidden_states)

    def project_logits(self, latent_states):
        return self.output_proj(latent_states)

    def sample_block_tokens(
        self, *, hidden_states, first_prev_token_ids, temperature=0.0
    ):
        from .dlite import _sample_sequential_tokens

        logits = self.project_logits(self.hidden_proj(hidden_states))
        shape = logits.shape[:-1]
        ids = _sample_sequential_tokens(
            logits.reshape(-1, logits.size(-1)), temperature
        )
        return ids.reshape(shape), logits


@dataclass
class Evidence:
    values: torch.Tensor  # B,T,H; raw frozen Target hidden
    positions: torch.Tensor  # B,T; source text-sequence positions
    valid: torch.Tensor  # B,T
    decode_start: torch.Tensor  # B; prompt lengths
    state: torch.Tensor = None  # 1,B,R; accepted prefix only (inference)
    state_prefix: torch.Tensor = None  # B,T,H; teacher-forced prefix (training)
    state_override: torch.Tensor = None  # diagnostic-only B,N,R
    reset_state: bool = False
    encoded: torch.Tensor = None  # inference-only normalized static bank
    pooled: object = None  # fixed-slot inference cache
    projected: object = None  # inference-only per-layer evidence K/V


def gather_evidence(
    states, draft, input_ids, prompt_mask, target_config, attention_mask=None
):
    from litespec.core.dlite import _hidden_at_layer

    cfg = draft.config.dlite_config
    raw = _hidden_at_layer(
        states, cfg["evidence_layer_id"], draft.config.num_target_layers
    )
    visual = input_ids == target_config.image_token_id
    video_id = getattr(target_config, "video_token_id", None)
    if video_id is not None:
        visual |= input_ids == video_id
    mode = cfg["context_mode"]
    valid = (
        attention_mask.bool()
        if attention_mask is not None
        else torch.ones_like(input_ids, dtype=torch.bool)
    )
    if mode != "pivot_history":
        valid = valid & prompt_mask & visual
    # Compact visual evidence without truncating or averaging distinct positions.
    width = (
        max(1, int(valid.sum(-1).max()))
        if mode != "pivot_history"
        else input_ids.size(1)
    )
    rows, locs, keeps = [], [], []
    positions = torch.arange(input_ids.size(1), device=input_ids.device)
    for b in range(input_ids.size(0)):
        idx = positions if mode == "pivot_history" else positions[valid[b]]
        n = idx.numel()
        rows.append(F.pad(raw[b, idx], (0, 0, 0, width - n)))
        locs.append(F.pad(idx, (0, width - n)))
        keeps.append(F.pad(valid[b, idx], (0, width - n), value=False))
    return Evidence(
        torch.stack(rows),
        torch.stack(locs),
        torch.stack(keeps),
        prompt_mask.sum(-1).long(),
        state_prefix=raw,
    )


class VisualKVModule(nn.Module):
    def __init__(self, hidden_size, config):
        super().__init__()
        self.mode = config["context_mode"]
        self.norm = nn.LayerNorm(hidden_size)
        self.slots = int(config.get("visual_slots", 16))
        self.rank = int(config.get("visual_state_rank", 64))
        if self.slots < 1 or self.rank < 1:
            raise ValueError("visual_slots and visual_state_rank must be positive")
        if self.dynamic:
            self.query = nn.Linear(hidden_size, self.slots * self.rank, bias=False)
            self.key = nn.Linear(hidden_size, self.rank, bias=False)
            self.state_query = nn.Linear(self.rank, self.slots * self.rank, bias=False)
            if self.mode == "visual_static_slots":
                self.slot_queries = nn.Parameter(
                    torch.randn(self.slots, self.rank) / math.sqrt(self.rank)
                )
        if self.mode == "visual_recurrent":
            self.state_input = nn.Linear(hidden_size, self.rank, bias=False)
            self.transition = nn.GRU(self.rank, self.rank, batch_first=True)

    @property
    def dynamic(self):
        return self.mode in {
            "visual_static_slots",
            "visual_pivot_conditioned",
            "visual_recurrent",
        }

    def states_at_anchors(self, evidence, anchors):
        bsz, blocks = anchors.shape
        if self.mode != "visual_recurrent" or evidence.reset_state:
            return evidence.values.new_zeros(bsz, blocks, self.rank)
        if evidence.state_override is not None:
            if evidence.state_override.shape != (bsz, blocks, self.rank):
                raise ValueError("state override shape differs from anchors")
            return evidence.state_override
        if evidence.state_prefix is None:
            s = evidence.state
            if s is None:
                s = evidence.values.new_zeros(1, bsz, self.rank)
            return s[0, :, None].expand(-1, blocks, -1)
        # Fused GRU over teacher-forced answer tokens, then gather at anchor-1.
        # Prefix independence makes this identical to any accepted-token chunking.
        rows = []
        for b in range(bsz):
            start = int(evidence.decode_start[b])
            raw = evidence.state_prefix[b : b + 1, start:]
            if raw.size(1):
                states, _ = self.transition(self.state_input(raw))
                states = F.pad(states, (0, 0, start, 0))
            else:
                states = raw.new_zeros(1, start, self.rank)
            idx = (anchors[b] - 1).clamp(min=0, max=states.size(1) - 1)
            rows.append(states[0, idx])
        return torch.stack(rows)

    def prepare(self, evidence, pivot, anchors):
        if evidence.pooled is not None:
            values, positions, valid = evidence.pooled
            return values, positions, valid, False
        values = (
            evidence.encoded
            if evidence.encoded is not None
            else self.norm(evidence.values)
        )
        bsz, blocks = anchors.shape
        if not self.dynamic:
            return values, evidence.positions, evidence.valid, False
        q = (
            self.slot_queries[None, None].expand(bsz, blocks, -1, -1)
            if self.mode == "visual_static_slots"
            else self.query(pivot.mean(2)).view(bsz, blocks, self.slots, self.rank)
        )
        state = self.states_at_anchors(evidence, anchors)
        q = q + self.state_query(state).view_as(q)
        keys = self.key(values)
        scores = torch.einsum("bnmr,btr->bnmt", q, keys) / math.sqrt(self.rank)
        allowed = evidence.valid[:, None, None, :] & (
            evidence.positions[:, None, None, :] < anchors[:, :, None, None] - 1
        )
        scores = scores.float().masked_fill(~allowed, torch.finfo(torch.float32).min)
        weights = scores.softmax(-1) * allowed
        slots = torch.einsum("bnmt,bth->bnmh", weights.to(values.dtype), values)
        keep = allowed.any(-1).expand(-1, -1, self.slots).reshape(bsz, -1)
        return slots.reshape(bsz, blocks * self.slots, -1), None, keep, True

    def commit(self, evidence, raw):
        if self.mode == "visual_recurrent" and not evidence.reset_state and raw.size(1):
            _, evidence.state = self.transition(self.state_input(raw), evidence.state)
            evidence.state = evidence.state.detach()


def visibility(
    anchors,
    block_keep,
    bank_positions,
    bank_valid,
    bank_length,
    slots_per_block,
    chs,
    width,
):
    """One closure shared by dense CPU and sparse CUDA masks. No future history."""
    blocks = anchors.shape[1]

    def mask(b, h, q, k):
        block = (q // width).clamp(max=blocks - 1)
        safe = k.clamp(max=max(bank_length - 1, 0))
        if slots_per_block:
            evidence = (
                (k < bank_length)
                & ((k // slots_per_block) == block)
                & bank_valid[b, safe]
            )
        else:
            evidence = (
                (k < bank_length)
                & bank_valid[b, safe]
                & (bank_positions[b, safe] < anchors[b, block] - 1)
            )
        pivot = (
            (k >= bank_length)
            & (k < bank_length + blocks * chs)
            & (((k - bank_length) // chs) == block)
        )
        local = (k >= bank_length + blocks * chs) & (
            ((k - bank_length - blocks * chs) // width) == block
        )
        return (
            (evidence | pivot | local)
            & block_keep[b, block]
            & (q < blocks * width)
            & (k < bank_length + blocks * (chs + width))
        )

    return mask


def forward_context(
    draft, *, query, pivot, anchors, block_keep, evidence=None, kernel_options=None
):
    bsz, blocks = anchors.shape
    width, chs = draft.draft_query_length, draft.chs_num_layers
    query = query.reshape(bsz, blocks * width, -1)
    pivot_context = draft._apply_chs_depth_embedding(pivot).reshape(
        bsz, blocks * chs, -1
    )
    if evidence is None:
        bank = query.new_empty(bsz, 0, query.size(-1))
        bank_pos = anchors.new_zeros(bsz, 1)
        bank_valid = torch.zeros(bsz, 1, dtype=torch.bool, device=query.device)
        grouped = False
    else:
        bank, bank_pos, bank_valid, grouped = draft.visual_kv.prepare(
            evidence, pivot, anchors
        )
    length = bank.size(1)
    context = torch.cat([bank, pivot_context], 1)
    mask_fn = visibility(
        anchors,
        block_keep,
        bank_pos,
        bank_valid,
        length,
        draft.visual_kv.slots if grouped else 0,
        chs,
        width,
    )
    if draft.config._attn_implementation == "flex_attention":
        from .flex_attention import create_flex_block_mask

        mask = create_flex_block_mask(
            mask_fn,
            B=bsz,
            H=None,
            Q_LEN=blocks * width,
            KV_LEN=context.size(1) + blocks * width,
            device=query.device,
            backend="TRITON",
        )
    else:
        q = torch.arange(blocks * width, device=query.device)[None, None, :, None]
        k = torch.arange(context.size(1) + blocks * width, device=query.device)[
            None, None, None, :
        ]
        b = torch.arange(bsz, device=query.device)[:, None, None, None]
        allowed = mask_fn(b, 0, q, k)
        mask = query.new_zeros(allowed.shape).masked_fill(
            ~allowed, torch.finfo(query.dtype).min
        )
    # Identity rotation for extra evidence; exact DLite local positions for pivot/block.
    local = (
        torch.arange(1, width + 1, device=query.device)
        .repeat(blocks)[None]
        .expand(bsz, -1)
    )
    pos = torch.cat([anchors.new_zeros(bsz, length + blocks * chs), local], -1)
    embeddings = draft.rotary_emb(query, pos)
    for index, layer in enumerate(draft.layers):
        prefix = (
            evidence.projected[index]
            if evidence is not None and evidence.projected is not None and not grouped
            else None
        )
        query = layer(
            hidden_states=query,
            target_hidden=context,
            attention_mask=mask,
            position_embeddings=embeddings,
            preprojected_context=prefix,
            kernel_options=kernel_options,
        )
    return draft.norm(query)
