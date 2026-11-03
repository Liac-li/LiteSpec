"""Numerical and integration checks for recurrent DLite physical layers."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from litespec.modeling.draft.dlite import DLiteDraftModel
from litespec.vlm.benchmark import RequestPositionState
from litespec.vlm.static_batch import generate_static_batch
from test_vlm import draft_config, tiny_qwen35
from test_vlm_static_batch import example_for
from test_vlm_aligned import fixture_model
from run_dlite_loop import MODELS, main, position_counts, report


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)
    torch.manual_seed(42)


def model(loops=5, layers=1):
    config = draft_config()
    config.num_hidden_layers = layers
    config.layer_types = ["full_attention"]*layers
    config.dlite_config["num_loops"] = loops
    return DLiteDraftModel(config)


def inputs(draft, batch=2, blocks=2):
    anchors = torch.arange(5, 5+blocks)[None].expand(batch, -1)
    context_pos, query_pos = draft.build_block_position_ids(
        anchors, anchors.unsqueeze(-1), torch.ones(batch, blocks, 1, dtype=torch.bool))
    return dict(position_ids=query_pos, rotary_position_ids=torch.cat([context_pos, query_pos], 1),
                noise_embedding=torch.randn(batch, blocks*draft.block_size, draft.config.hidden_size),
                target_hidden=torch.randn(batch, blocks, draft.chs_num_layers, draft.config.hidden_size),
                history_hidden=torch.empty(batch, blocks, 0, draft.config.hidden_size))


def reference_forward(draft, values, loops):
    # Independent reference: original layer path recomputes context QKV per loop.
    context = draft._fuse_target_hidden(values["target_hidden"], values["history_hidden"])
    hidden = values["noise_embedding"]
    rotary = draft.rotary_emb(hidden, values["rotary_position_ids"])
    for _ in range(loops):
        for layer in draft.layers:
            hidden = layer(hidden_states=hidden, target_hidden=context, attention_mask=None,
                           position_ids=values["position_ids"], position_embeddings=rotary)
    return draft.norm(hidden)


@pytest.mark.parametrize("loops,layers", [(1, 5), (3, 1), (5, 1), (3, 2)])
def test_shared_projection_matches_unrolled_outputs_and_gradients(loops, layers):
    recurrent = model(loops, layers)
    reference = deepcopy(recurrent)
    a = inputs(recurrent)
    b = {k: v.clone() for k, v in a.items()}
    a["target_hidden"].requires_grad_()
    b["target_hidden"].requires_grad_()
    actual = recurrent(**a)
    expected = reference_forward(reference, b, loops)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    weights = torch.randn_like(actual)
    (actual*weights).sum().backward()
    (expected*weights).sum().backward()
    torch.testing.assert_close(a["target_hidden"].grad, b["target_hidden"].grad, rtol=5e-4, atol=2e-5)
    for (name, param), (_, other) in zip(recurrent.named_parameters(), reference.named_parameters()):
        if other.grad is None:
            assert param.grad is None, name
        else:
            torch.testing.assert_close(param.grad, other.grad, rtol=5e-4, atol=2e-5, msg=name)


def test_context_projects_once_but_block_qkv_changes_each_loop():
    draft = model()
    attention = draft.layers[0].self_attn
    lengths, handles = [], []
    for projection in (attention.k_proj, attention.v_proj):
        handles.append(projection.register_forward_pre_hook(lambda m, args: lengths.append(args[0].shape[1])))
    values = inputs(draft)
    draft(**values)
    for handle in handles:
        handle.remove()
    assert lengths.count(4) == 2  # one K and one V projection of four CHS slots
    assert lengths.count(8) == 10  # fresh K/V for eight query slots on five loops


def test_reload_keeps_loop_depth_and_old_configs_default_to_one(tmp_path):
    draft = model(5).eval()
    values = inputs(draft)
    draft.save_pretrained(tmp_path)
    loaded = DLiteDraftModel.from_pretrained(tmp_path, attn_implementation="sdpa").eval()
    assert loaded.num_loops == 5 and len(loaded.layers) == 1
    torch.testing.assert_close(draft(**values), loaded(**values), rtol=0, atol=0)
    config = draft_config()
    assert "num_loops" not in config.dlite_config
    legacy = DLiteDraftModel(config).eval()
    old_input = inputs(legacy)
    assert legacy.num_loops == 1
    torch.testing.assert_close(legacy(**old_input), reference_forward(legacy, old_input, 1), rtol=0, atol=0)


@pytest.mark.parametrize("value", [0, -1, True, 2.5, "5"])
def test_invalid_loop_config(value):
    with pytest.raises(ValueError, match="num_loops"):
        model(value)


def test_fixed_depth_override_is_bounded():
    draft = model(5)
    values = inputs(draft)
    torch.testing.assert_close(draft(**values, num_loops=3), reference_forward(draft, values, 3))
    for value in (0, 6, True, 2.5):
        with pytest.raises(ValueError, match="num_loops"):
            draft(**values, num_loops=value)


@pytest.mark.parametrize("loops", [3, 5])
def test_real_image_training_backward_includes_shared_layer_and_rnn(loops):
    from litespec.vlm import train_aligned
    target, draft, _, example = fixture_model()
    config = draft.config
    config.dlite_config["num_loops"] = loops
    draft = DLiteDraftModel(config)
    cfg = dict(block_size=4, num_anchors=4, loss_decay_gamma=4,
               final_ce_weight=0.1, l1_loss_weight=1.0, base_lm_ce_weight=0.0,
               flex_attention_backend="TRITON", algorithm="dlite")
    online = train_aligned.make_online(target, draft, cfg)
    kwargs = train_aligned.prepare_forward(target, online, example)
    output = online(**kwargs)
    output.loss.backward()
    assert torch.isfinite(output.loss)
    for param in (draft.layers[0].self_attn.k_proj.weight, draft.sequential_head.output_proj.weight):
        assert torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
    assert all(param.grad is None for param in target.parameters())


@pytest.mark.parametrize("loops", [1, 3, 5])
@pytest.mark.parametrize("image", [False, True])
@pytest.mark.parametrize("state_mode", ["full_target_replay", "per_token"])
def test_qwen35_text_image_and_batch32(loops, image, state_mode):
    target = tiny_qwen35().requires_grad_(False)
    draft = model(loops).eval()
    example = example_for(target, image)
    results = []
    for batch in (1, 32):
        with RequestPositionState(target).activate():
            result = generate_static_batch(target, draft, example, batch_size=batch, max_new_tokens=6,
                                           state_mode=state_mode)
        assert result["replicas_identical"]
        assert result["output_tokens"] == batch*len(result["token_ids"])
        assert len(result["proposal_spans"]) == len(result["accept_lengths"])
        results.append(result)
    assert results[0]["token_ids"] == results[1]["token_ids"]
    with RequestPositionState(target).activate():
        ar = generate_static_batch(target, None, example, max_new_tokens=6)
    assert results[0]["token_ids"] == ar["token_ids"]


def test_launcher_creates_three_controls_with_identical_training_settings(tmp_path, capsys):
    data = tmp_path/"train.jsonl"
    data.write_text('{"question":"example","answer":"answer"}\n')
    target = tmp_path/"target"
    target.mkdir()
    run = tmp_path/"run"
    main(["train", "--run-dir", str(run), "--train-data", str(data), "--target-model", str(target),
          "--gpus", "4,5", "--dry-run"])
    from litespec.vlm.train_aligned import load_config
    configs = [load_config(run/name/"config.yaml") for name in MODELS]
    assert [(c["num_draft_layers"], c["num_loops"]) for c in configs] == [(5, 1), (1, 5), (1, 3)]
    for key in ("seed", "num_epochs", "chs_num_layers", "sequential_rank", "batch_size", "block_size",
                "accumulation_steps", "final_ce_weight", "l1_loss_weight", "train_data"):
        assert len({c[key] for c in configs}) == 1
    text = capsys.readouterr().out
    assert text.count('"--reuse-validation-from"') == 2
    assert text.count('"--max-samples", "20000"') == 3
    with pytest.raises(ValueError, match="fresh"):
        main(["train", "--run-dir", str(run)])


def test_position_statistics_exclude_anchor_and_censored_positions():
    rows = [dict(result=dict(batch_size=32, accept_lengths=[2, 1, 3], proposal_spans=[3, 0, 2]))]
    hits, trials = position_counts(rows, 3)
    assert hits == [64, 32, 0]
    assert trials == [64, 64, 32]


def test_evaluation_dispatches_three_checkpoints_and_reports(tmp_path, capsys):
    run = tmp_path/"trained"
    run.mkdir()
    (run/"experiment.json").write_text(json.dumps(dict(target_model="tiny")))
    for label, (layers, loops) in zip(MODELS, ((5, 1), (1, 5), (1, 3))):
        checkpoint = run/label/"last"
        checkpoint.mkdir(parents=True)
        (checkpoint/"config.json").write_text(json.dumps(dict(num_hidden_layers=layers, dlite_config=dict(num_loops=loops))))
    text = tmp_path/"gsm8k.jsonl"
    text.write_text('{"turns":["question"]}\n')
    data = tmp_path/"data"
    data.mkdir()
    (data/"textvqa_test.jsonl").write_text('{}\n')
    main(["eval", "--run-dir", str(run), "--datasets", "gsm8k", "textvqa",
          "--text-data", str(text), "--data-dir", str(data), "--dry-run"])
    commands = capsys.readouterr().out
    for label in MODELS:
        assert commands.count(f'"--checkpoint", "{label}"') == 2
    assert commands.count('"--batch-sizes", "1", "32"') == 2
    assert commands.count('"--profile-decode"') == 2
    assert commands.count('"--reference-input"') == 1  # vision only; GSM8K uses existing text reader
    result = run/"evaluation"/"gsm8k"
    result.mkdir()
    summaries = []
    for label in ("ar", *MODELS):
        summaries.append(dict(model=label, batch_size=32, mean_accepted_draft_tokens=1,
            mean_committed_length=2, output_tokens_per_second=100, raw_throughput_speedup_vs_ar=2,
            matched=0, requests=32, peak_allocated_gib=1))
        records = [dict(result=dict(batch_size=32, accept_lengths=[2], proposal_spans=[3]))]
        (result/f"{label}_b32.jsonl").write_text(json.dumps(records[0])+"\n")
    (result/"summary.json").write_text(json.dumps(summaries))
    report(run/"evaluation")
    comparison = (run/"evaluation/comparison.md").read_text()
    assert "0/32" in comparison and "Raw ratio" in comparison
    positions = json.loads((run/"evaluation/position_acceptance.json").read_text())
    assert len(positions) == 21
    assert positions[0]["prefix_acceptance"] == 1
    assert positions[1]["prefix_acceptance"] == 0
    assert positions[6]["prefix_acceptance"] is None
