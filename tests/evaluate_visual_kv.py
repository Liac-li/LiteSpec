"""Matched frozen snapshots; raw rates are reported separately from AR match."""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from evaluate_memory_ablation import run_logged, report
from run_vlm_matrix import select_data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litespec.modeling.draft.visual_kv import STATIC_EXPERIMENTS, EXPERIMENTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-group", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    group = args.run_group.expanduser().resolve()
    root = args.output_dir or group / "visual_kv_evaluation"
    names = os.environ.get("EXPERIMENTS", " ".join(STATIC_EXPERIMENTS)).split()
    if len(names) != len(set(names)):
        raise ValueError("duplicate experiments")
    budget = int(os.environ.get("IMAGE_MAX_TOKENS", "1024"))
    target = Path(
        os.environ.get("TARGET_MODEL", str(Path.home() / "LMM/lmm_model/Qwen3.5-4B"))
    ).expanduser()
    models = []
    for name in names:
        if name not in EXPERIMENTS:
            raise ValueError(f"unknown experiment: {name}")
        path = group / name / "last"
        cfg = json.loads((path / "config.json").read_text())
        if cfg["dlite_config"].get("experiment_name") != name:
            raise ValueError(f"checkpoint name mismatch: {path}")
        if cfg["vlm_config"].get("image_max_tokens") != budget:
            raise ValueError("training/evaluation image budget mismatch")
        models.append((name, path))
    tests = os.environ.get("TESTS", "text-only charxiv_val textvqa").split()
    data = select_data(
        SimpleNamespace(
            tests=tests,
            samples=int(os.environ.get("EVAL_SAMPLES", "64")),
            seed=int(os.environ.get("SAMPLE_SEED", "42")),
        )
    )
    vision = json.loads((target / "config.json").read_text())["vision_config"]
    unit = vision["patch_size"] * vision["spatial_merge_size"]
    root.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("EVAL_GPU", "0")}
    rows = []
    for case, packet in data.items():
        snapshot = root / f"{case}.jsonl"
        snapshot.write_text(packet["serialized"])
        common = [
            sys.executable,
            str(Path(__file__).with_name("benchmark_vlm_static_batch.py")),
            "--target-model",
            str(target),
            "--data",
            str(snapshot),
            "--dataset-name",
            case,
            "--max-samples",
            str(packet["samples"]),
            "--max-new-tokens",
            os.environ.get("EVAL_MAX_NEW_TOKENS", "256"),
            "--batch-sizes",
            *os.environ.get("BATCH_SIZES", "1").split(),
            "--image-max-pixels",
            str(budget * unit * unit),
            "--cpu-inputs",
            "--state-mode",
            os.environ.get("STATE_MODE", "per_token"),
        ]
        dest = root / f"{case}_correct"
        command = [*common, "--output-dir", str(dest)]
        for name, path in models:
            command.extend(["--checkpoint", name, str(path)])
        if os.environ.get("PROFILE_DECODE", "0") == "1":
            command.extend(
                [
                    "--profile-decode",
                    "--profile-samples",
                    os.environ.get("PROFILE_SAMPLES", "4"),
                ]
            )
        run_logged(command, env, dest)
        rows.extend(json.loads((dest / "summary.json").read_text()))
        for name, path in models:
            conditions = []
            if (
                case != "gsm8k"
                and EXPERIMENTS[name][0] not in {"pivot", "pivot_history"}
                and os.environ.get("VISUAL_DIAGNOSTICS", "1") == "1"
            ):
                conditions += [
                    ("disabled", ["--memory-condition", "disabled"]),
                    ("wrong_image", ["--memory-condition", "wrong_image"]),
                ]
            if name == "visual_recurrent":
                conditions.append(("reset", ["--reset-visual-state"]))
            for condition, flags in conditions:
                out = root / f"{case}_{name}_{condition}"
                run_logged(
                    [
                        *common,
                        "--ar-reference",
                        str(dest),
                        "--checkpoint",
                        f"{name}_{condition}",
                        str(path),
                        "--output-dir",
                        str(out),
                        *flags,
                    ],
                    env,
                    out,
                )
                rows.extend(
                    r
                    for r in json.loads((out / "summary.json").read_text())
                    if r["model"] != "ar"
                )
        report(root, rows)
    print(f"Summary: {root/'comparison.csv'}")


if __name__ == "__main__":
    main()
