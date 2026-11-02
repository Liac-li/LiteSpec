"""Replicated-prompt static batches with decode-only timing and hybrid rollback.

Attention KV is allocated once during prefill. Linear-attention state is kept in
the Transformers hybrid cache. Rejection snapshots recurrent matrices, crops KV
and convolution history, and replays the committed prefix; it never clones the
entire attention history. This is offline HF batching, not a serving scheduler.
"""
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from litespec.modeling.draft.dlite import (
    _begin_recurrent_cache_transaction, _restore_recurrent_cache_transaction,
    _crop_target_cache,
)
from .runtime import clock, draft_block, selected_layers
from .context import ContextPolicy


class PreallocatedAttentionLayer(DynamicLayer):
    """Fixed storage, variable visible prefix; supports HF 5.x relative crop."""
    is_compileable = False

    def __init__(self, capacity):
        super().__init__()
        self.capacity = int(capacity)
        self.length = 0

    def update(self, key_states, value_states, *args, **kwargs):
        end = self.length + key_states.shape[-2]
        if end > self.capacity:
            raise ValueError(f"target KV capacity exceeded: {end} > {self.capacity}")
        if not self.is_initialized:
            self.dtype, self.device = key_states.dtype, key_states.device
            self.keys = key_states.new_empty(*key_states.shape[:-2], self.capacity, key_states.shape[-1])
            self.values = value_states.new_empty(*value_states.shape[:-2], self.capacity, value_states.shape[-1])
            self.is_initialized = True
        self.keys[..., self.length:end, :].copy_(key_states)
        self.values[..., self.length:end, :].copy_(value_states)
        self.length = end
        return self.keys[..., :end, :], self.values[..., :end, :]

    def get_seq_length(self):
        return self.length

    def get_mask_sizes(self, query_length):
        return self.length + int(query_length), 0

    def get_max_length(self):
        return self.capacity

    def crop(self, tokens_to_remove):
        if tokens_to_remove > 0:
            raise ValueError("relative crop requires a nonpositive removal count")
        self.length = max(0, self.length + int(tokens_to_remove))

    def reset(self):
        self.length = 0


def make_hybrid_cache(target, capacity):
    """Keep native GDN layers; replace only full-attention DynamicLayers."""
    cache = DynamicCache(config=target.config)
    count = 0
    for index, layer in enumerate(cache.layers):
        if type(layer) is DynamicLayer:
            cache.layers[index] = PreallocatedAttentionLayer(capacity)
            count += 1
    if not count:
        raise ValueError("static benchmark requires full-attention layers in a hybrid Qwen3.5 target")
    if not hasattr(cache, "activate_past_recording"):
        raise RuntimeError("static hybrid benchmark requires Transformers hybrid-cache transaction support")
    return cache


def replicate_inputs(example, batch_size):
    if batch_size < 1 or example.inputs["input_ids"].shape != (1, example.prefill_length):
        raise ValueError("replicate one unpadded prompt with a positive batch size")
    result = {}
    for key, value in example.inputs.items():
        if torch.is_tensor(value) and value.ndim:
            if key == "position_ids" and value.ndim == 3:
                result[key] = value.repeat(1, batch_size, 1)
            else:
                # Image patches/grid rows are image-major, not batch-major.
                # Repeating the whole tensor matches B identical image blocks.
                result[key] = value.repeat(batch_size, *([1] * (value.ndim - 1)))
        elif isinstance(value, list):
            result[key] = value * batch_size
        else:
            result[key] = value
    return result


@torch.no_grad()
def generate_static_batch(target, draft, example, *, batch_size=1, max_new_tokens=128,
                          verification="block", profile=False, state_mode="full_target_replay",
                          memory_example=None, disable_memory=False, reset_visual_state=False,
                          context_schedule="always", history_scope="full"):
    from .gdn_state import STATE_MODES, GDNStateTransaction
    if state_mode not in STATE_MODES:
        raise ValueError(f"unknown GDN state mode: {state_mode}")
    if state_mode != "full_target_replay" and verification != "block":
        raise ValueError("GDN state modes require block verification")
    if state_mode != "full_target_replay" and draft is not None:
        with GDNStateTransaction(target, state_mode) as transaction:
            return _generate_static_batch(target, draft, example, batch_size=batch_size,
                max_new_tokens=max_new_tokens, verification=verification, profile=profile,
                transaction=transaction, state_mode=state_mode,
                memory_example=memory_example, disable_memory=disable_memory, reset_visual_state=reset_visual_state,
                context_schedule=context_schedule, history_scope=history_scope)
    return _generate_static_batch(target, draft, example, batch_size=batch_size,
        max_new_tokens=max_new_tokens, verification=verification, profile=profile, state_mode=state_mode,
        memory_example=memory_example, disable_memory=disable_memory, reset_visual_state=reset_visual_state,
        context_schedule=context_schedule, history_scope=history_scope)


def _generate_static_batch(target, draft, example, *, batch_size, max_new_tokens,
                           verification, profile, state_mode, transaction=None,
                           memory_example=None, disable_memory=False, reset_visual_state=False,
                           context_schedule="always", history_scope="full"):
    """One true batch of identical greedy requests; no request-by-request loop."""
    from .baselines import DraftCache, is_baseline
    if max_new_tokens < 1 or verification not in {"block", "sequential"}:
        raise ValueError("positive output cap and block/sequential verification required")
    baseline = draft is not None and is_baseline(draft)
    if draft is not None and not baseline and not draft.is_student:
        raise ValueError("static batch DLite benchmark requires a history-free student")
    if baseline:
        policy = getattr(draft.config, "vlm_config", {}).get("context_policy")
        if policy and (policy["prefill_text"] != "keep" or policy["prefill_vision"] != "keep" or policy["decode_mode"] != "all"):
            raise ValueError("cached baseline benchmark requires full history")
    inputs = replicate_inputs(example, batch_size)
    device = inputs["input_ids"].device
    anchor = example.prefill_length
    width = draft.block_size if draft is not None else 1
    eos = getattr(target.config, "text_config", target.config).eos_token_id
    eos = eos if isinstance(eos, (list, tuple)) else [eos]
    eos_ids = inputs["input_ids"].new_tensor([value for value in eos if value is not None])
    started = clock(device)  # Replication/CPU image preparation excluded.
    cache = make_hybrid_cache(target, anchor + max_new_tokens + width)
    memory = None
    memory_build_seconds = 0.
    donor_prefill_seconds = 0.
    reader = getattr(draft, "memory_reader", None)
    visual_control = draft is not None and getattr(draft, "visual_kv", None) is not None
    if (context_schedule != "always" or history_scope != "full") and not visual_control and draft is not None:
        raise ValueError("context schedules require a trained visual-KV/history interface")
    visual_cache = None
    visual_donor = None
    if visual_control and memory_example is not None:
        if disable_memory or draft.config.dlite_config['context_mode'] in {'pivot', 'pivot_history'}:
            raise ValueError('wrong-image diagnosis requires an enabled visual-only context')
        donor_inputs = replicate_inputs(memory_example, batch_size)
        donor_started = clock(device)
        donor_output = target(**donor_inputs, use_cache=False, output_hidden_states=True, logits_to_keep=1)
        visual_donor = (donor_output, donor_inputs['input_ids'])
        donor_prefill_seconds = clock(device)-donor_started
    if memory_example is not None and not visual_control:
        if reader is None or draft.config.dlite_config.get("memory_mode") != "vision" or disable_memory:
            raise ValueError("wrong-image memory requires an enabled vision-memory checkpoint")
        from .memory import build_prompt_memory
        donor_started = clock(device)
        donor_inputs = replicate_inputs(memory_example, batch_size)
        donor = target(**donor_inputs, use_cache=False, output_hidden_states=True, logits_to_keep=1)
        memory_started = clock(device)
        memory = reader.project(build_prompt_memory(donor.hidden_states, draft,
            donor_inputs["input_ids"], torch.ones_like(donor_inputs["input_ids"], dtype=torch.bool), target.config))
        memory_build_seconds = clock(device) - memory_started
        donor_prefill_seconds = clock(device) - donor_started
        del donor, donor_inputs
    output = target(**inputs, past_key_values=cache, use_cache=True, logits_to_keep=1,
                    output_hidden_states=draft is not None)
    cache = output.past_key_values
    pending = output.logits[:, -1:].argmax(-1)
    pivot = (selected_layers(output.hidden_states, draft.target_layer_ids,
                             draft.config.num_target_layers)[:, -1:].detach().clone()
             if draft is not None else None)
    history = DraftCache(draft, target, output, anchor) if baseline else None
    if reader is not None and not disable_memory and memory is None:
        from .memory import build_prompt_memory
        memory_started = clock(device)
        memory = reader.project(build_prompt_memory(output.hidden_states, draft,
            inputs["input_ids"], torch.ones_like(inputs["input_ids"], dtype=torch.bool), target.config))
        memory_build_seconds = clock(device) - memory_started
    if visual_control:
        from .visual_kv import VisualKVCache
        memory_started = clock(device)
        visual_cache = VisualKVCache(draft, target, output, inputs, donor=visual_donor,
            disable=disable_memory, reset_state=reset_visual_state,
            context_schedule=context_schedule, history_scope=history_scope)
        memory_build_seconds = clock(device)-memory_started
        visual_donor = None
        if memory_example is not None:
            del donor_output, donor_inputs
    memory_tokens = (int(visual_cache.evidence.valid[0].sum())
        if visual_cache is not None and visual_cache.evidence is not None else
        int(memory.valid[0].sum()) if memory is not None else 0)
    del output, inputs
    prefill_end = clock(device)
    timeline = None
    if profile:
        from .inference_profile import DecodeTimeline
        timeline = DecodeTimeline(device)
    blocks, lengths, emitted, replays, target_calls = [], [], 0, 0, 0
    peak_snapshot_bytes = 0
    round_diagnostics = []
    while emitted < max_new_tokens:
        if draft is None:
            if timeline: timeline.mark("emit")
            candidates, kept = pending, 1
        else:
            if timeline: timeline.mark("draft")
            if visual_control:
                proposals, _ = visual_cache.propose(draft, target, pending, pivot, anchor)
            elif baseline:
                proposals, _ = history.propose(draft, target, pending, anchor)
            else:
                hidden, _ = draft_block(draft, target.get_input_embeddings(), anchor_ids=pending,
                    pivot=pivot, history=None, anchor=anchor, prefill_length=example.prefill_length,
                    vision_mask=example.vision_mask, policy=ContextPolicy.no_history(),
                    prompt_memory=memory, disable_memory=disable_memory, timeline=timeline)
                proposals = draft.sample_draft_tokens(draft_hidden=hidden,
                    first_prev_token_ids=pending[:, 0], temperature=0.0)[0]
            candidates = torch.cat([pending, proposals], 1)[:, :max_new_tokens-emitted]
            if verification == "sequential":
                if timeline: timeline.mark("target_verify")
                # Correctness control only, still batched across B replicas.
                states, predictions = [], []
                verified_hits = []
                for index in range(candidates.shape[1]):
                    output = target(input_ids=candidates[:, index:index+1], past_key_values=cache,
                                    use_cache=True, output_hidden_states=True, logits_to_keep=1)
                    cache = output.past_key_values
                    target_calls += 1
                    states.append(output.hidden_states)
                    predictions.append(output.logits[:, -1:].argmax(-1))
                    if index + 1 < candidates.shape[1]:
                        verified_hits.append(bool(torch.equal(candidates[:, index+1:index+2], predictions[-1])))
                    if index + 1 == candidates.shape[1] or not torch.equal(candidates[:, index+1:index+2], predictions[-1]):
                        break
                kept = len(states)
                pending = predictions[-1]
                features = tuple(torch.cat(parts, 1) for parts in zip(*states))
                output.hidden_states = features
            else:
                if timeline: timeline.mark("state_snapshot")
                snapshot = _begin_recurrent_cache_transaction(cache) if transaction is None else None
                if snapshot:
                    peak_snapshot_bytes = max(peak_snapshot_bytes, sum(t.numel()*t.element_size()
                        for _, record in snapshot for t in record.values()))
                if transaction is None and snapshot is None and not cache.is_croppable:
                    raise RuntimeError("target hybrid cache has no supported recurrent-state rollback")
                if timeline: timeline.mark("target_verify")
                if transaction is None:
                    output = target(input_ids=candidates, past_key_values=cache, use_cache=True,
                                    output_hidden_states=True)
                else:
                    with transaction.capture(cache):
                        output = target(input_ids=candidates, past_key_values=cache, use_cache=True,
                                        output_hidden_states=True)
                cache = output.past_key_values
                target_calls += 1
                if timeline: timeline.mark("acceptance")
                predictions = output.logits.argmax(-1)
                match = candidates[:, 1:] == predictions[:, :-1]
                verified_hits = match[0].detach().clone()
                accepted = match.cumprod(1).sum(1)
                kept = int(accepted.min()) + 1
                if transaction is not None:
                    if timeline: timeline.mark("state_select" if state_mode == "per_token" else "gdn_state_replay")
                    transaction.commit(cache, kept, candidates.shape[1])
                    if timeline: timeline.mark("cache_commit")
                    _crop_target_cache(cache, current_length=anchor+candidates.shape[1],
                                       tokens_to_remove=candidates.shape[1]-kept)
                    pending = predictions[:, kept-1:kept]
                elif kept < candidates.shape[1] and snapshot is not None:
                    if timeline: timeline.mark("cache_restore")
                    _crop_target_cache(cache, current_length=anchor+candidates.shape[1],
                                       tokens_to_remove=candidates.shape[1])
                    _restore_recurrent_cache_transaction(snapshot)
                    if timeline: timeline.mark("target_replay")
                    output = target(input_ids=candidates[:, :kept], past_key_values=cache,
                                    use_cache=True, output_hidden_states=True, logits_to_keep=1)
                    cache = output.past_key_values
                    replays += 1
                    target_calls += 1
                    if timeline: timeline.mark("cache_commit")
                    pending = output.logits[:, -1:].argmax(-1)
                    _crop_target_cache(cache, current_length=anchor+kept, tokens_to_remove=0)
                else:
                    if timeline: timeline.mark("cache_commit")
                    _crop_target_cache(cache, current_length=anchor+candidates.shape[1],
                                       tokens_to_remove=candidates.shape[1]-kept)
                    pending = predictions[:, kept-1:kept]
                del snapshot
            if timeline: timeline.mark("hidden_features")
            pivot = selected_layers(output.hidden_states, draft.target_layer_ids,
                                    draft.config.num_target_layers)[:, kept-1:kept].detach().clone()
            if visual_control:
                visual_cache.commit(draft, output, kept, anchor)
            if baseline:
                history.commit(draft, output, kept)
            del output
            round_diagnostics.append(dict(
                round_index=len(round_diagnostics), decode_offset=emitted,
                first_candidate_hit=verified_hits[0] if len(verified_hits) else None,
                verified_candidate_hits=verified_hits, committed_length=kept,
                candidate_count=candidates.shape[1]-1,
                # Later block hits condition on candidate prefixes, not the AR trace.
                comparison_scope="candidate_conditioned_target"))
            anchor += kept
            if timeline: timeline.mark("emit")
        # Replicas use the same EOS boundary; assert actual equality after timing.
        end_positions = torch.isin(candidates[0, :kept], eos_ids).nonzero()
        stop = end_positions.numel() > 0
        actual = int(end_positions[0, 0]) + 1 if stop else kept
        blocks.append(candidates[:, :actual].clone())
        emitted += actual
        if draft is not None:
            lengths.append(actual)
        if stop or emitted == max_new_tokens:
            break
        if draft is None:
            if timeline: timeline.mark("ar_target_decode")
            output = target(input_ids=pending, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            pending = output.logits[:, -1:].argmax(-1)
            target_calls += 1
            del output
    if timeline: timeline.mark(None)
    finished = clock(device)
    # Materialize diagnostic booleans only AFTER the timed CUDA interval.
    for row in round_diagnostics:
        if torch.is_tensor(row["verified_candidate_hits"]):
            row["verified_candidate_hits"] = row["verified_candidate_hits"].tolist()
        row["first_candidate_hit"] = row["verified_candidate_hits"][0] if row["verified_candidate_hits"] else None
    tokens = torch.cat(blocks, 1)
    if not torch.equal(tokens, tokens[:1].expand_as(tokens)):
        raise RuntimeError("greedy replicas diverged within a batch; lockstep replicated-input benchmark assumptions failed")
    result = dict(token_ids=tokens[0].tolist(), batch_size=batch_size,
        output_tokens=batch_size * emitted, accept_lengths=lengths,
        prefill_tokens=example.prefill_length,
        visual_tokens_with_delimiters=int(example.vision_mask.sum()),
        prefill_seconds=prefill_end-started, decode_seconds=finished-prefill_end,
        total_seconds=finished-started, target_decode_calls=target_calls,
        recovery_replays=replays, replicas_identical=True,
        round_diagnostics=round_diagnostics, context_schedule=context_schedule, history_scope=history_scope,
        cache_policy="preallocated_attention_kv_recurrent_snapshot_and_prefix_replay",
        verification=verification, state_mode=state_mode,
        gdn_verify_backend="native_hf" if transaction is None else
            ("triton_fused_recurrent" if device.type == "cuda" else "cpu_reference_recurrent"),
        peak_transaction_record_bytes=transaction.peak_record_bytes if transaction else peak_snapshot_bytes)
    result.update(memory_tokens=memory_tokens, memory_build_seconds=memory_build_seconds,
                  donor_prefill_seconds=donor_prefill_seconds,
                  memory_condition="disabled" if disable_memory else "wrong_image" if memory_example is not None else "correct")
    if visual_control:
        result.update(experiment_name=draft.config.dlite_config['experiment_name'],
            reset_visual_state=reset_visual_state,
            retained_evidence_tokens=int(visual_cache.evidence.valid[0].sum()) if visual_cache.evidence is not None else 0,
            draft_parameters=sum(p.numel() for p in draft.parameters()))
    if transaction:
        result["cache_policy"] = "preallocated_attention_kv_" + state_mode
    if timeline:
        result['profile'] = timeline.resolve(result['decode_seconds'])
    return result

