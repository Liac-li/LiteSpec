import json
from pathlib import Path
import pytest
import torch
import yaml
from litespec.vlm import train_aligned as aligned
from litespec.modeling.draft.visual_kv import experiment_settings, EXPERIMENTS
from litespec.vlm.baselines import load_draft
from test_vlm_aligned import Processor
from test_vlm import tiny_qwen35


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_actual_runner_two_steps_export_and_resume(name, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(
        aligned,
        "load_target",
        lambda *a, **kw: (tiny_qwen35().requires_grad_(False), Processor()),
    )
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(dict(id=str(i), question="describe", answer="a square"))
            for i in range(4)
        )
    )
    output = tmp_path / name
    cfg = dict(
        upstream_config="scripts/config/dlite_sft.yaml",
        target_model="tiny",
        train_data=str(manifest),
        output_dir=str(output),
        device="cpu",
        model_role="student",
        report_to="none",
        log_interval=1,
        save_interval=4000,
        num_draft_layers=1,
        chs_num_layers=2,
        sequential_rank=8,
        block_size=4,
        num_anchors=3,
        num_epochs=1,
        batch_size=1,
        accumulation_steps=1,
        use_muon=False,
        visual_slots=2,
        visual_state_rank=4,
        **experiment_settings(name),
    )
    config = tmp_path / (name + ".yaml")
    config.write_text(yaml.safe_dump(cfg))
    arguments = ["--config", str(config), "--max-steps", "2", "--save-training-state"]
    aligned.run(arguments)
    assert (output / "last/config.json").exists()
    draft = load_draft(output / "last", attn_implementation="flash_attention_2")
    assert draft.config._attn_implementation == "sdpa"
    before = {k: v.clone() for k, v in draft.state_dict().items()}
    aligned.run([*arguments, "--resume", str(output / "last")])
    after = load_draft(output / "last").state_dict()
    for key, value in before.items():
        torch.testing.assert_close(value, after[key], rtol=0, atol=0)


def test_unknown_experiment_fails_before_model_loading(tmp_path):
    cfg = dict(upstream_config="scripts/config/dlite_sft.yaml", experiment_name="A")
    config = tmp_path / "bad.yaml"
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="unknown visual-KV"):
        aligned.load_config(config)


def test_fixed_prefix_probe_writes_repairs_and_wrong_state(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from PIL import Image
    from test_visual_kv import model

    spec = importlib.util.spec_from_file_location(
        "visual_probe", Path(__file__).with_name("probe_visual_kv.py")
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    torch.set_num_threads(1)
    torch.manual_seed(3)
    target = tiny_qwen35().eval().requires_grad_(False)
    draft = model("visual_recurrent").eval()
    monkeypatch.setattr(probe, "load_target", lambda *a, **kw: (target, Processor()))
    monkeypatch.setattr(probe, "load_draft", lambda *a, **kw: draft)
    import litespec.vlm.memory

    monkeypatch.setattr(
        litespec.vlm.memory, "configure_image_tokens", lambda *a, **kw: None
    )
    records = []
    for i in range(2):
        image = tmp_path / f"{i}.png"
        Image.new("RGB", (8, 8), (i * 100, 0, 0)).save(image)
        records.append(dict(id=str(i), question="describe", image=str(image)))
    data = tmp_path / "eval.jsonl"
    data.write_text("\n".join(json.dumps(x) for x in records))
    out = tmp_path / "probe"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--target-model",
            "tiny",
            "--checkpoint",
            "tiny",
            "--data",
            str(data),
            "--output-dir",
            str(out),
            "--samples",
            "2",
            "--max-new-tokens",
            "8",
            "--device",
            "cpu",
        ],
    )
    probe.main()
    rows = [json.loads(x) for x in (out / "anchors.jsonl").read_text().splitlines()]
    assert rows
    assert {"wrong_image", "wrong_state", "reset_state", "disabled", "correct"} == set(
        rows[0]["conditions"]
    )
    assert "repairs" in rows[0]["conditions"]["correct"]
