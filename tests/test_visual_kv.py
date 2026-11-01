"""Causality, conditioning and committed-state tests for the experiment matrix."""

from dataclasses import replace
import pytest
import torch
from litespec.modeling.draft.dlite import DLiteDraftModel
from litespec.modeling.draft.visual_kv import EXPERIMENTS, experiment_settings
from litespec.vlm import train_aligned as aligned
from litespec.vlm.visual_kv import VisualKVCache
from litespec.vlm.static_batch import generate_static_batch
from test_vlm_aligned import fixture_model, CONFIG, Processor
from test_vlm import draft_config, tiny_qwen35, tiny_image_inputs
from litespec.vlm.runtime import PreparedExample, selected_layers, visual_positions


def model(name):
    cfg = draft_config()
    cfg.dlite_config.update(
        experiment_settings(name),
        evidence_layer_id=2,
        visual_slots=2,
        visual_state_rank=4,
    )
    return DLiteDraftModel(cfg)


def setup(name):
    torch.set_num_threads(1)
    target, _, _, example = fixture_model()
    draft = model(name)
    cfg = aligned.load_config(CONFIG)
    cfg.update(block_size=4, num_anchors=3)
    online = aligned.make_online(target, draft, cfg)
    kwargs = aligned.prepare_forward(target, online, example)
    return target, draft, online, example, kwargs


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_all_modes_train_roundtrip_and_parallel_previous_token_independence(
    name, tmp_path
):
    target, draft, online, example, kwargs = setup(name)
    result = online(**kwargs)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert draft.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert not any(p.grad is not None for p in target.parameters())
    if name == "visual_recurrent":
        assert draft.visual_kv.transition.weight_ih_l0.grad.abs().sum() > 0
    batch = kwargs["prepared_batch"]
    z = online.forward_backbone(batch, seq_len=kwargs["seq_len"])
    if draft.config.dlite_config["prediction_head"] == "parallel":
        a = draft.sequential_head.forward_teacher_forcing(
            hidden_states=z, prev_token_ids=batch.prev_token_ids
        )
        b = draft.sequential_head.forward_teacher_forcing(
            hidden_states=z, prev_token_ids=torch.zeros_like(batch.prev_token_ids)
        )
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    directory = tmp_path / name
    draft.save_pretrained(directory)
    loaded = DLiteDraftModel.from_pretrained(directory, attn_implementation="sdpa")
    assert loaded.config.dlite_config["experiment_name"] == name
    for k, v in draft.state_dict().items():
        torch.testing.assert_close(v, loaded.state_dict()[k], rtol=0, atol=0)


@pytest.mark.parametrize(
    "name", ["pivot_history_rnn", "visual_recurrent", "visual_pivot_conditioned"]
)
def test_future_target_features_cannot_change_first_block(name):
    _, draft, online, _, kwargs = setup(name)
    batch = kwargs["prepared_batch"]
    evidence = batch.context_evidence
    a = int(batch.anchor_positions[0, 0])
    expected = online.forward_backbone(batch, seq_len=kwargs["seq_len"])[:, :1]
    altered = replace(
        evidence,
        values=evidence.values.clone(),
        state_prefix=evidence.state_prefix.clone(),
    )
    altered.values[evidence.positions >= a - 1] += 500
    altered.state_prefix[:, a - 1 :] += 500
    # Current pivot=a-1 is provided independently; recurrent state SHOULD include
    # it, so alter only a..end for the state channel.
    altered.state_prefix[:, a - 1] = evidence.state_prefix[:, a - 1]
    batch.context_evidence = altered
    actual = online.forward_backbone(batch, seq_len=kwargs["seq_len"])[:, :1]
    torch.testing.assert_close(expected, actual)


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_training_inference_first_anchor_agree(name):
    target, draft, online, example, kwargs = setup(name)
    batch = kwargs["prepared_batch"]
    anchor = int(batch.anchor_positions[0, 0])
    output = target(**example.inputs, use_cache=False, output_hidden_states=True)
    inputs = {
        **example.inputs,
        "input_ids": example.inputs["input_ids"][:, : example.prefill_length],
    }
    # Gather only prompt source for visual cache; full history through pivot.
    prefix_output = type(
        "Output",
        (),
        {"hidden_states": tuple(x[:, :anchor] for x in output.hidden_states)},
    )()
    prefix_ids = example.inputs["input_ids"][:, :anchor]
    cache = VisualKVCache(draft, target, prefix_output, {"input_ids": prefix_ids})
    if name == "visual_recurrent":
        # Cache construction receives prompt, then commits teacher tokens to anchor.
        prompt_out = type(
            "Output",
            (),
            {
                "hidden_states": tuple(
                    x[:, : example.prefill_length] for x in output.hidden_states
                )
            },
        )()
        cache = VisualKVCache(draft, target, prompt_out, inputs)
        commit_out = type(
            "Output",
            (),
            {
                "hidden_states": tuple(
                    x[:, example.prefill_length : anchor] for x in output.hidden_states
                )
            },
        )()
        cache.commit(
            draft, commit_out, anchor - example.prefill_length, example.prefill_length
        )
    pivot = selected_layers(output.hidden_states, draft.target_layer_ids, 4)[
        :, anchor - 1 : anchor
    ]
    pending = example.inputs["input_ids"][:, anchor : anchor + 1]
    ids = pending.new_full((1, 4), draft.mask_token_id)
    ids[:, :1] = pending
    query = draft.build_inference_query_embeddings(target.get_input_embeddings(), ids)
    prediction = draft(
        noise_embedding=query,
        target_hidden=pivot,
        position_ids=ids,
        context_anchors=pending.new_tensor([[anchor]]),
        context_block_keep=torch.ones(1, 1, dtype=torch.bool),
        context_evidence=cache.evidence,
    )[:, -3:]
    expected = online.forward_backbone(batch, seq_len=kwargs["seq_len"])[:, 0]
    torch.testing.assert_close(prediction, expected, rtol=2e-4, atol=2e-5)


def test_recurrent_commit_is_independent_of_accepted_chunk_sizes():
    _, draft, online, _, kwargs = setup("visual_recurrent")
    bank = kwargs["prepared_batch"].context_evidence
    raw = bank.state_prefix[:, 8:]
    whole, chunks = replace(bank, state=None), replace(bank, state=None)
    draft.visual_kv.commit(whole, raw)
    for a, b in [(0, 1), (1, 4), (4, raw.size(1))]:
        draft.visual_kv.commit(chunks, raw[:, a:b])
    torch.testing.assert_close(whole.state, chunks.state)


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_real_tiny_hybrid_image_decode_matches_ar(name):
    torch.manual_seed(7)
    torch.set_num_threads(1)
    target = tiny_qwen35().eval()
    draft = model(name).eval()
    inputs = tiny_image_inputs()
    example = PreparedExample(
        inputs,
        inputs["input_ids"].size(1),
        visual_positions(inputs["input_ids"][0], target.config),
        "tiny",
    )
    ar = generate_static_batch(target, None, example, max_new_tokens=7)
    result = generate_static_batch(
        target, draft, example, max_new_tokens=7, state_mode="per_token"
    )
    assert result["token_ids"] == ar["token_ids"]
    assert result["experiment_name"] == name


def test_pivot_rnn_control_matches_existing_dlite_backbone():
    target, legacy, old_online, example = fixture_model()
    experiment = model("pivot_rnn")
    experiment.load_state_dict(legacy.state_dict(), strict=False)
    new_online = aligned.make_online(
        target,
        experiment,
        dict(aligned.load_config(CONFIG), block_size=4, num_anchors=3),
    )
    kwargs = aligned.prepare_forward(target, old_online, example)
    batch = kwargs["prepared_batch"]
    expected = old_online.forward_backbone(batch, seq_len=kwargs["seq_len"])
    actual = new_online.forward_backbone(batch, seq_len=kwargs["seq_len"])
    torch.testing.assert_close(expected, actual)
