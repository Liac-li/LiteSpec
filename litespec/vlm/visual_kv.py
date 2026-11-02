"""Training and committed-only inference adapters for the visual-KV controls."""

import torch
from litespec.core.dlite import OnlineDLiteModel
from litespec.modeling.draft.visual_kv import Evidence, gather_evidence


def attach_context(batch, states, draft, example, target_config):
    from .memory import prompt_mask_from_example

    if draft.config.dlite_config["context_mode"] == "pivot":
        batch.context_evidence = None
    else:
        batch.context_evidence = gather_evidence(
            states,
            draft,
            example.inputs["input_ids"],
            prompt_mask_from_example(example),
            target_config,
            example.inputs.get("attention_mask"),
        )


class OnlineVisualKV(OnlineDLiteModel):
    def forward_backbone(self, batch, *, seq_len):
        d = self.draft_model
        output = d(
            noise_embedding=batch.query_embeddings,
            target_hidden=batch.target_hidden,
            position_ids=batch.token_position_ids.flatten(1),
            context_anchors=batch.anchor_positions,
            context_block_keep=batch.block_keep_mask,
            context_evidence=getattr(batch, "context_evidence", None),
            kernel_options=self.flex_kernel_options,
        )
        bsz, blocks = batch.anchor_positions.shape
        return output.view(bsz, blocks, d.draft_query_length, -1)[
            :, :, -d.proposal_length :
        ]


class VisualKVCache:
    """Per-request evidence and recurrent state. Never stored on the model instance."""

    def __init__(
        self,
        draft,
        target,
        output,
        inputs,
        *,
        donor=None,
        disable=False,
        reset_state=False,
        state_donor=None,
        context_schedule="always",
        history_scope="full",
    ):
        self.mode = draft.config.dlite_config["context_mode"]
        if context_schedule not in {"always", "off", "first_round"}:
            raise ValueError("unknown context schedule")
        if history_scope not in {"full", "prefill"}:
            raise ValueError("unknown history scope")
        if history_scope != "full" and self.mode != "pivot_history":
            raise ValueError("prefill-only scope requires a trained pivot_history interface")
        self.context_schedule = context_schedule
        self.history_scope = history_scope
        self.rounds = 0
        self.evidence = None
        if self.mode == "pivot" or disable or context_schedule == "off":
            return
        source, ids = donor if donor is not None else (output, inputs["input_ids"])
        prompt = torch.ones_like(ids, dtype=torch.bool)
        self.evidence = gather_evidence(
            source.hidden_states, draft, ids, prompt, target.config
        )
        if donor is not None:
            # Donor prompt length must not hide its already-known visual tokens.
            self.evidence.positions = torch.zeros_like(self.evidence.positions)
        self.evidence.state_prefix = None
        self.evidence.reset_state = reset_state
        self.evidence.encoded = draft.visual_kv.norm(self.evidence.values)
        if self.mode == "visual_static_slots":
            bsz = ids.size(0)
            dummy = self.evidence.values.new_zeros(
                bsz, 1, draft.chs_num_layers, draft.config.hidden_size
            )
            anchors = ids.new_full((bsz, 1), ids.size(1))
            values, _, valid, _ = draft.visual_kv.prepare(self.evidence, dummy, anchors)
            self.evidence.pooled = (values, ids.new_zeros(values.shape[:2]), valid)
        self._project(draft)
        if state_donor is not None:
            self.evidence.state = state_donor.detach().clone()

    def _project(self, draft):
        if self.evidence is None or (
            draft.visual_kv.dynamic and self.mode != "visual_static_slots"
        ):
            return
        x = (
            self.evidence.pooled[0]
            if self.evidence.pooled is not None
            else self.evidence.encoded
        )
        self.evidence.projected = [
            (layer.self_attn.k_proj(x), layer.self_attn.v_proj(x))
            for layer in draft.layers
        ]

    def propose(self, draft, target, pending, pivot, anchor):
        ids = pending.new_full((pending.size(0), draft.block_size), draft.mask_token_id)
        ids[:, :1] = pending
        query = draft.build_inference_query_embeddings(
            target.get_input_embeddings(), ids
        )
        anchors = pending.new_full((pending.size(0), 1), anchor)
        output = draft(
            noise_embedding=query,
            target_hidden=pivot,
            position_ids=pending.new_zeros(pending.size(0), draft.block_size),
            context_anchors=anchors,
            context_block_keep=torch.ones_like(anchors, dtype=torch.bool),
            context_evidence=self.evidence,
        )
        hidden = draft._prediction_hidden(output)
        result = draft.sample_draft_tokens(
            draft_hidden=hidden, first_prev_token_ids=pending[:, 0], temperature=0.0
        )
        self.rounds += 1
        if self.context_schedule == "first_round":
            # No backward graph in inference. Drop raw bank AND per-layer K/V;
            # commit must not append history or update state after this release.
            self.evidence = None
        return result

    def commit(self, draft, output, kept, anchor):
        if self.evidence is None or (self.mode == "pivot_history" and self.history_scope == "prefill"):
            return
        from litespec.core.dlite import _hidden_at_layer

        cfg = draft.config.dlite_config
        raw = _hidden_at_layer(
            output.hidden_states,
            cfg["evidence_layer_id"],
            draft.config.num_target_layers,
        )[:, :kept].detach()
        draft.visual_kv.commit(self.evidence, raw)
        if self.mode == "pivot_history" and self.history_scope == "full":
            bank = self.evidence
            start = anchor
            positions = torch.arange(start, start + kept, device=raw.device)[
                None
            ].expand(raw.size(0), -1)
            encoded = draft.visual_kv.norm(raw)
            bank.values = torch.cat([bank.values, raw], 1)
            bank.positions = torch.cat([bank.positions, positions], 1)
            bank.valid = torch.cat(
                [bank.valid, torch.ones_like(positions, dtype=torch.bool)], 1
            )
            bank.encoded = torch.cat([bank.encoded, encoded], 1)
            bank.projected = [
                (
                    torch.cat([old[0], layer.self_attn.k_proj(encoded)], 1),
                    torch.cat([old[1], layer.self_attn.v_proj(encoded)], 1),
                )
                for old, layer in zip(bank.projected, draft.layers)
            ]

