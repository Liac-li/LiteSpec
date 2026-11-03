"""Recurrent DFlash: tied-unroll equivalence, cache isolation and VLM integration."""
from copy import deepcopy
import json

import pytest
import torch
from transformers import DynamicCache

from litespec.modeling.draft.dflash import DFlashDraftModel
from litespec.vlm import baselines, train_aligned
from litespec.vlm.benchmark import RequestPositionState
from litespec.vlm.static_batch import generate_static_batch
from test_vlm import draft_config, tiny_qwen35
from test_vlm_aligned import Processor
from test_vlm_static_batch import example_for
from run_dlite_loop import main, MODEL_SPECS


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)
    torch.manual_seed(42)


def model(loops=5, layers=1):
    cfg = draft_config()
    del cfg.dlite_config
    cfg.architectures = ["DFlashDraftModel"]
    cfg.num_hidden_layers = layers
    cfg.layer_types = ["full_attention"] * layers
    cfg.dflash_config = dict(target_layer_ids=[0, 3], mask_token_id=127, num_loops=loops)
    return DFlashDraftModel(cfg)


def unroll(draft, loops):
    cfg = deepcopy(draft.config)
    cfg.num_hidden_layers *= loops
    cfg.layer_types *= loops
    cfg.dflash_config["num_loops"] = 1
    reference = DFlashDraftModel(cfg)
    weights = draft.state_dict()
    for name in reference.state_dict():
        source = name
        if name.startswith("layers."):
            parts = name.split(".")
            parts[1] = str(int(parts[1]) % len(draft.layers))
            source = ".".join(parts)
        reference.state_dict()[name].copy_(weights[source])
    return reference


@pytest.mark.parametrize("loops,layers", [(1, 5), (3, 1), (5, 1), (3, 2)])
def test_unrolled_outputs_and_accumulated_shared_gradients(loops, layers):
    draft = model(loops, layers)
    reference = unroll(draft, loops)
    noise = torch.randn(2, 4, 32)
    context = torch.randn(2, 3, 64)
    values = dict(noise_embedding=noise, target_hidden=context, position_ids=torch.arange(7)[None])
    actual, expected = draft(**values), reference(**values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    for name, param in draft.named_parameters():
        if name.startswith("layers."):
            parts = name.split(".")
            index = int(parts[1])
            gradients = []
            for repeat in range(loops):
                parts[1] = str(repeat * layers + index)
                gradients.append(reference.get_parameter(".".join(parts)).grad)
            expected_grad = sum(gradients)
        else:
            expected_grad = reference.get_parameter(name).grad
        torch.testing.assert_close(param.grad, expected_grad, rtol=5e-4, atol=2e-5, msg=name)


@pytest.mark.parametrize("loops,layers", [(3, 1), (5, 1), (3, 2)])
def test_cached_rounds_match_unrolled_reference_and_crop_every_virtual_layer(loops, layers):
    draft = model(loops, layers).eval()
    reference = unroll(draft, loops).eval()
    caches = [DynamicCache(), DynamicCache()]
    for start, context_length, anchor in [(0, 6, 6), (6, 2, 8), (8, 1, 9)]:
        values = dict(noise_embedding=torch.randn(2, 4, 32),
                      target_hidden=torch.randn(2, context_length, 64),
                      position_ids=torch.arange(start, anchor + 4)[None], use_cache=True)
        with torch.no_grad():
            actual = draft(**values, past_key_values=caches[0])
            expected = reference(**values, past_key_values=caches[1])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for cache in caches:
            assert len(cache.layers) == loops * layers
            assert all(layer.get_seq_length() == anchor + 4 for layer in cache.layers)
            cache.crop(-4)
            assert all(layer.get_seq_length() == anchor for layer in cache.layers)
        for a, b in zip(caches[0].layers, caches[1].layers):
            torch.testing.assert_close(a.keys, b.keys, rtol=0, atol=0)
            torch.testing.assert_close(a.values, b.values, rtol=0, atol=0)


@pytest.mark.parametrize("loops", [3, 5])
def test_image_objective_backward_export_and_reload(loops, tmp_path):
    target = tiny_qwen35().requires_grad_(False)
    cfg = train_aligned.load_config("scripts/config/vlm/dflash_loop_1l3r.yaml")
    cfg.update(num_loops=loops, block_size=4, num_anchors=2, target_layer_ids=[0, 3])
    draft = baselines.make_draft(target, Processor(), cfg)
    online = baselines.make_online(target, draft, cfg)
    example = example_for(target, True)
    # Processor supplies a longer supervised answer than the evaluation prompt.
    from litespec.vlm.runtime import PreparedExample
    inputs = Processor().apply_chat_template([], add_generation_prompt=False)
    training = PreparedExample(inputs, 8, torch.zeros(inputs["input_ids"].shape[1], dtype=torch.bool), "train")
    output = online(**train_aligned.prepare_forward(target, online, training))
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert draft.layers[0].self_attn.k_proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in target.parameters())
    draft.save_pretrained(tmp_path)
    loaded = baselines.load_draft(tmp_path, attn_implementation="sdpa").eval()
    assert loaded.num_loops == loops and len(loaded.layers) == 1
    assert loaded.target_layer_ids == [0, 3]
    with RequestPositionState(target).activate():
        assert generate_static_batch(target, loaded, example, max_new_tokens=6)["replicas_identical"]


@pytest.mark.parametrize("image", [False, True])
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("loops", [3, 5])
@pytest.mark.parametrize("state_mode", ["full_target_replay", "per_token"])
def test_qwen35_cached_loop_text_image_and_batch32(image, batch_size, loops, state_mode):
    target = tiny_qwen35().requires_grad_(False)
    draft = model(loops).eval()
    example = example_for(target, image)
    with RequestPositionState(target).activate():
        ar = generate_static_batch(target, None, example, batch_size=batch_size, max_new_tokens=7)
    with RequestPositionState(target).activate():
        result = generate_static_batch(target, draft, example, batch_size=batch_size,
                                       max_new_tokens=7, state_mode=state_mode)
    assert result["token_ids"] == ar["token_ids"]
    assert result["replicas_identical"]


def test_legacy_config_and_bounded_fixed_exit(tmp_path):
    from benchmark_vlm_static_batch import set_inference_loops
    draft = model(5)
    draft.save_pretrained(tmp_path)
    before = (tmp_path / "config.json").read_bytes()
    loaded = baselines.load_draft(tmp_path, attn_implementation="sdpa")
    set_inference_loops(loaded, 3)
    assert loaded.num_loops == 3 and loaded.config.dflash_config["num_loops"] == 5
    assert (tmp_path / "config.json").read_bytes() == before
    legacy = deepcopy(draft.config)
    legacy.dflash_config.pop("num_loops")
    assert DFlashDraftModel(legacy).num_loops == 1
    for bad in (0, 6, True, 2.5):
        with pytest.raises(ValueError, match="inference loops"):
            set_inference_loops(loaded, bad)


def test_benchmark_cli_records_dflash_fixed_exits(monkeypatch, tmp_path):
    import sys
    import benchmark_vlm_static_batch as cli
    target = tiny_qwen35().requires_grad_(False)
    checkpoint = tmp_path / "draft"
    model(5).save_pretrained(checkpoint)
    original = (checkpoint / "config.json").read_bytes()
    data = tmp_path / "eval.jsonl"
    data.write_text(json.dumps(dict(id="one", question="describe")) + "\n")
    monkeypatch.setattr(cli, "load_target", lambda *a, **kw: (target, Processor()))
    arguments = ["benchmark", "--target-model", "tiny", "--data", str(data),
        "--output-dir", str(tmp_path / "results"), "--batch-sizes", "1", "32",
        "--max-samples", "1", "--max-new-tokens", "4", "--warmup", "0", "--device", "cpu"]
    for depth in (1, 3, 5):
        arguments += ["--checkpoint", f"exit{depth}", str(checkpoint),
                      "--checkpoint-loops", f"exit{depth}", str(depth)]
    monkeypatch.setattr(sys, "argv", arguments)
    cli.main()
    summaries = json.loads((tmp_path / "results/summary.json").read_text())
    assert len(summaries) == 8
    for row in summaries:
        if row["model"] != "ar":
            assert row["trained_num_loops"] == 5 and row["physical_layers"] == 1
            assert row["inference_num_loops"] == int(row["model"][-1])
    assert (checkpoint / "config.json").read_bytes() == original


@pytest.mark.parametrize("value", [0, -1, True, 2.5, "5"])
def test_invalid_trained_loop_depth(value):
    with pytest.raises(ValueError, match="num_loops"):
        model(value)


def test_six_model_five_epoch_plan_and_all_fixed_exits(tmp_path, capsys):
    data, target, run = tmp_path / "train.jsonl", tmp_path / "target", tmp_path / "run"
    data.write_text('{}\n')
    target.mkdir()
    main(["train", "--run-dir", str(run), "--train-data", str(data), "--target-model", str(target),
          "--families", "dlite", "dflash", "--epochs", "5", "--gpus", "0,1,2,3,4,5,6,7", "--dry-run"])
    manifest = json.loads((run / "experiment.json").read_text())
    assert manifest["models"] == list(MODEL_SPECS)
    assert manifest["num_epochs"] == 5 and manifest["global_batch"] == 32
    configs = [train_aligned.load_config(run / label / "config.yaml") for label in MODEL_SPECS]
    assert all(c["num_epochs"] == 5 and c["block_size"] == 8 for c in configs)
    assert [(c["num_draft_layers"], c["num_loops"]) for c in configs] == [(5, 1), (1, 5), (1, 3)] * 2
    assert all(c["target_context_layers"] == 5 for c in configs[3:])
    assert capsys.readouterr().out.count('"--reuse-validation-from"') == 5
    # Simulate the checkpoint metadata of a completed training group.
    manifest["dry_run"] = False
    (run / "experiment.json").write_text(json.dumps(manifest))
    for label, (family, layers, loops) in MODEL_SPECS.items():
        checkpoint = run / label / "last"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(json.dumps(dict(num_hidden_layers=layers,
            **{family + "_config": dict(num_loops=loops)})))
    main(["eval", "--run-dir", str(run), "--datasets", "gsm8k", "--text-data", str(data),
          "--gpus", "0,1,2,3,4,5", "--dry-run"])
    settings = json.loads((run / "evaluation/experiment.json").read_text())
    assert len(settings["exits"]) == 18
    assert [e["inference_num_loops"] for e in settings["exits"]] == [1, 1, 2, 3, 4, 5, 1, 2, 3] * 2
    command = json.loads((run / "evaluation/queue_status.json").read_text())["jobs"][0]["command"]
    assert command.count("--checkpoint") == command.count("--checkpoint-loops") == 18
