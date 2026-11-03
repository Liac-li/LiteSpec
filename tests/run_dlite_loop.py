"""Three DLite controls: common-data training and B=1/32 static evaluation.

Run with the existing .venv-vlm Python. This does not install dependencies.
"""
import argparse
from collections import deque
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("dlite_5l1r", "dlite_1l5r", "dlite_1l3r")
VISION_DATASETS = ("charxiv_val", "flickr30k", "textvqa", "hr_bench_4k", "hr_bench_8k")


def execute(command, log_path, env, *, dry_run=False):
    print("RUN " + json.dumps([str(x) for x in command]), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen([str(x) for x in command], cwd=ROOT, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            process.terminate()
            process.wait()
            raise


def environment(gpus):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, PYTHONUNBUFFERED="1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def evaluation_gpu_ids(value):
    items = [item.strip() for item in value.split(",")]
    if not items or any(not item.isdecimal() for item in items):
        raise ValueError("evaluation GPUs must be distinct nonnegative IDs separated by commas")
    ids = [str(int(item)) for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation GPUs must be distinct nonnegative IDs separated by commas")
    return ids


def run_evaluation_jobs(jobs, gpus, result_dir, *, dry_run=False):
    """Run one independent dataset process per GPU; parent owns queue status.

    Every dataset task includes its own same-device AR and all fixed exits.
    Children exclusively own their dataset result directories and log files.
    """
    pending, active = deque(jobs), {}
    states = {name: dict(dataset=name, status="pending", gpu=None, pid=None, returncode=None,
                         command=[str(item) for item in command]) for name, command in jobs}
    status_path = result_dir/"queue_status.json"

    def save_status():
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(gpus=gpus, dry_run=dry_run, jobs=list(states.values())), indent=2))
        temporary.replace(status_path)

    if dry_run:
        for index, (name, command) in enumerate(jobs):
            gpu = gpus[index % len(gpus)]
            states[name].update(status="planned", gpu=gpu)
            print(f"PLAN {name}: GPU={gpu}", flush=True)
            print("RUN " + json.dumps([str(item) for item in command]), flush=True)
        save_status()
        return list(states.values())

    def stop_process(process):
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()

    save_status()
    try:
        while pending or active:
            for gpu in gpus:
                if gpu in active or not pending:
                    continue
                name, command = pending.popleft()
                log_path = result_dir/f"{name}.log"
                log = log_path.open("w")
                try:
                    process = subprocess.Popen([str(item) for item in command], cwd=ROOT,
                        env=environment(gpu), stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=os.name == "posix")
                except BaseException as error:
                    log.close()
                    states[name].update(status="failed", gpu=gpu, error=str(error))
                    save_status()
                    raise
                active[gpu] = (name, process, log)
                states[name].update(status="running", gpu=gpu, pid=process.pid, log=str(log_path))
                save_status()
                print(f"START {name}: GPU={gpu}, PID={process.pid}; log={log_path}", flush=True)
            completed = []
            for gpu, (name, process, log) in active.items():
                code = process.poll()
                if code is not None:
                    completed.append((gpu, name, log, code))
            failure = None
            for gpu, name, log, code in completed:
                log.close()
                del active[gpu]
                states[name].update(status="completed" if code == 0 else "failed", returncode=code)
                save_status()
                print(f"{'DONE' if code == 0 else 'FAILED'} {name}: GPU={gpu}, exit={code}", flush=True)
                if code and failure is None:
                    failure = subprocess.CalledProcessError(code, states[name]["command"])
            if failure is not None:
                raise failure
            if not completed and active:
                time.sleep(0.1)
    finally:
        for gpu, (name, process, log) in active.items():
            try:
                stop_process(process)
            finally:
                log.close()
                states[name].update(status="cancelled", returncode=process.poll())
        save_status()
    return list(states.values())


def train(args):
    if args.run_dir.exists():
        raise ValueError("choose a fresh --run-dir; existing experiments are never overwritten")
    source, target = args.train_data.expanduser().resolve(), args.target_model.expanduser().resolve()
    if not source.is_file() or not target.is_dir():
        raise ValueError("local --train-data and --target-model must exist")
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must contain distinct CUDA device IDs")
    args.run_dir.mkdir(parents=True)
    manifest = dict(target_model=str(target), train_data=str(source), max_samples=args.max_samples,
                    models=list(MODELS), num_epochs=1, block_size=8, seed=42,
                    chs_num_layers=12, sequential_rank=320, gpus=gpus,
                    micro_batch=args.micro_batch, accumulation=args.accumulation,
                    global_batch=len(gpus)*args.micro_batch*args.accumulation,
                    debug_steps=args.debug_steps, dry_run=args.dry_run)
    (args.run_dir/"experiment.json").write_text(json.dumps(manifest, indent=2))
    env = environment(args.gpus)
    first = args.run_dir/MODELS[0]
    for label in MODELS:
        config = yaml.safe_load((ROOT/"scripts/config/vlm"/f"{label.replace('dlite_', 'dlite_loop_')}.yaml").read_text())
        output = args.run_dir/label
        output.mkdir()
        config.update(target_model=str(target), train_data=str(source), output_dir=str(output),
                      batch_size=args.micro_batch, accumulation_steps=args.accumulation,
                      report_to=args.report_to)
        path = output/"config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                   f"--nproc-per-node={len(gpus)}", "-m", "litespec.vlm.train_aligned",
                   "--config", path, "--max-samples", str(args.max_samples)]
        if label != MODELS[0]:
            command += ["--reuse-validation-from", first]
        if args.debug_steps:
            command += ["--max-steps", str(args.debug_steps)]
        execute(command, output/"train.log", env, dry_run=args.dry_run)
        if not args.dry_run:
            effective = json.loads((output/"effective_config.json").read_text())
            reference = json.loads((first/"effective_config.json").read_text())
            for key in ("data_sha256", "eligible_indices_sha256", "eligible_samples", "total_optimizer_steps"):
                if effective[key] != reference[key]:
                    raise RuntimeError(f"training controls differ in {key}")
    print(f"Training group: {args.run_dir}", flush=True)


def evaluate(args):
    gpus = evaluation_gpu_ids(args.gpus if args.gpus is not None else args.gpu or "0")
    if args.gpu is not None and len(gpus) != 1:
        raise ValueError("--gpu selects one device; use --gpus for multiple devices")
    metadata = json.loads((args.run_dir/"experiment.json").read_text())
    if metadata.get("dry_run"):
        raise ValueError("this run contains dry-run plans, not trained checkpoints")
    target = args.target_model.expanduser().resolve() if args.target_model else Path(metadata["target_model"])
    result_dir = args.result_dir or args.run_dir/"evaluation"
    if result_dir.exists():
        raise ValueError("choose a fresh --result-dir; existing results are never overwritten")
    checkpoints = []
    loop_overrides = []
    exits = []
    for label in MODELS:
        checkpoint = args.run_dir/label/"last"
        if not (checkpoint/"config.json").is_file():
            raise FileNotFoundError(checkpoint)
        config = json.loads((checkpoint/"config.json").read_text())
        expected = dict(zip(MODELS, ((5, 1), (1, 5), (1, 3))))[label]
        if (config["num_hidden_layers"], config["dlite_config"].get("num_loops", 1)) != expected:
            raise ValueError(f"wrong checkpoint architecture for {label}")
        depths = range(1, expected[1]+1) if args.loop_sweep and expected[1] > 1 else [expected[1]]
        for depth in depths:
            alias = f"{label}_loop{depth}" if args.loop_sweep and expected[1] > 1 else label
            checkpoints += ["--checkpoint", alias, checkpoint]
            loop_overrides += ["--checkpoint-loops", alias, str(depth)]
            exits.append(dict(label=alias, checkpoint=str(checkpoint), trained_num_loops=expected[1],
                              inference_num_loops=depth, physical_layers=expected[0]))
    jobs = []
    if "gsm8k" in args.datasets:
        if not args.text_data.expanduser().is_file():
            raise FileNotFoundError(args.text_data)
        jobs.append(("gsm8k", args.text_data.expanduser().resolve(), "text"))
    from reference_vlm_data import dataset_path
    for name in args.datasets:
        if name != "gsm8k":
            jobs.append((name, dataset_path(args.data_dir.expanduser(), name), "vision"))
    # Validate all paths/checkpoints before loading the target for the first job.
    result_dir.mkdir(parents=True)
    (result_dir/"experiment.json").write_text(json.dumps(dict(
        training_run=str(args.run_dir), datasets=args.datasets, batch_sizes=args.batch_sizes,
        state_mode=args.state_mode, samples=args.samples, max_new_tokens=args.max_new_tokens,
        loop_sweep=args.loop_sweep, exits=exits, gpus=gpus,
        image_max_pixels=args.image_max_pixels, benchmark="static_replicated_batch"), indent=2))
    scheduled = []
    for name, data, mode in jobs:
        command = [sys.executable, "tests/benchmark_vlm_static_batch.py", "--target-model", target,
                   *checkpoints, *loop_overrides, "--data", data, "--dataset-name", name, "--data-split", "unspecified",
                   "--output-dir", result_dir/name, "--batch-sizes", *map(str, args.batch_sizes),
                   "--max-samples", str(args.samples), "--max-new-tokens", str(args.max_new_tokens),
                   "--warmup", "1", "--state-mode", args.state_mode, "--seed", "42", "--cpu-inputs"]
        if mode == "vision":
            command += ["--reference-input", "--input-mode", "vision", "--image-root", args.image_root.expanduser()]
        if args.image_max_pixels:
            command += ["--image-max-pixels", str(args.image_max_pixels)]
        if args.profile:
            command += ["--profile-decode", "--profile-samples", str(min(4, args.samples))]
        scheduled.append((name, command))
    try:
        run_evaluation_jobs(scheduled, gpus, result_dir, dry_run=args.dry_run)
    finally:
        # Read only completed datasets, never another child's live summary.
        status_path = result_dir/"queue_status.json"
        if not args.dry_run and status_path.is_file():
            completed = [job["dataset"] for job in json.loads(status_path.read_text())["jobs"]
                         if job["status"] == "completed"]
            if completed:
                report(result_dir, datasets=completed)
    print(f"Evaluation group: {result_dir}", flush=True)


def position_counts(records, width=7):
    hits, trials = [0]*width, [0]*width
    for row in records:
        result = row["result"]
        spans = result.get("proposal_spans")
        if spans is None or len(spans) != len(result["accept_lengths"]):
            raise ValueError("per-position report requires newly generated proposal_spans")
        for length, span in zip(result["accept_lengths"], spans):
            for index in range(width):
                if index < span:
                    trials[index] += result["batch_size"]
                    hits[index] += result["batch_size"] * int(index < length-1)
    return hits, trials


def report(root, datasets=None):
    rows, positions = [], []
    queue = root/"queue_status.json"
    if datasets is None and queue.is_file():
        status = json.loads(queue.read_text())
        if not status.get("dry_run"):
            datasets = [job["dataset"] for job in status["jobs"] if job["status"] == "completed"]
    for path in sorted(root.glob("*/summary.json")):
        if datasets is not None and path.parent.name not in datasets:
            continue
        summaries = json.loads(path.read_text())
        for source in summaries:
            row = dict(dataset=path.parent.name, **source)
            rows.append(row)
            if row["model"] == "ar":
                continue
            records = [json.loads(s) for s in (path.parent/f"{row['model']}_b{row['batch_size']}.jsonl").read_text().splitlines() if s.strip()]
            hits, trials = position_counts(records)
            for index, (hit, trial) in enumerate(zip(hits, trials)):
                positions.append(dict(dataset=path.parent.name, model=row["model"], batch_size=row["batch_size"],
                    inference_num_loops=row.get("inference_num_loops"),
                    draft_position=index, prefix_accepted=hit, eligible=trial,
                    prefix_acceptance=hit/trial if trial else None))
    if not rows:
        raise ValueError(f"no completed evaluation summaries in {root}")
    for name, values in (("comparison", rows), ("position_acceptance", positions)):
        (root/f"{name}.json").write_text(json.dumps(values, indent=2))
        if values:
            fields = list(dict.fromkeys(key for value in values for key in value))
            with (root/f"{name}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(values)
    lines = ["# DLite loop comparison", "", "Static replicated inputs; decode-only raw throughput ratios. "
             "B denotes batch size, not block length. Check token matches before claiming lossless speedup.", "",
             "Each loop label is a separate complete decoding run at that fixed depth, using the same trained checkpoint.", "",
             "| Dataset | Model | Layers | Loops | B | Accepted draft | Committed length | tok/s | Raw ratio / AR | Match | Peak GiB |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "—" if value is None else f"{value:.3f}"
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['model']} | {row.get('physical_layers', '—')} | "
            f"{row.get('inference_num_loops', '—')} | {row['batch_size']} | "
            f"{fmt(row['mean_accepted_draft_tokens'])} | {fmt(row['mean_committed_length'])} | "
            f"{row['output_tokens_per_second']:.2f} | {row['raw_throughput_speedup_vs_ar']:.3f} | "
            f"{row['matched']}/{row['requests']} | {fmt(row['peak_allocated_gib'])} |")
    lines += ["", "`position_acceptance.csv` reports prefix survival for draft positions @0–@6, "
              "excluding the known anchor; positions beyond the output cap or an emitted EOS are censored.",
              "Draft stage timings, when enabled, are measured in a separate instrumented pass; "
              "see each dataset's `inference_profile.md`."]
    (root/"comparison.md").write_text("\n".join(lines)+"\n")
    print(f"Comparison: {root/'comparison.md'}", flush=True)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    home = Path.home()
    workspace = ROOT.parent
    training = sub.add_parser("train")
    training.add_argument("--run-dir", type=Path, required=True)
    training.add_argument("--train-data", type=Path, default=workspace/"data/finevision_qwen35_sglang_20000/train.jsonl")
    training.add_argument("--target-model", type=Path, default=home/"LMM/lmm_model/Qwen3.5-4B")
    training.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"))
    training.add_argument("--micro-batch", type=positive, default=1)
    training.add_argument("--accumulation", type=positive, default=4)
    training.add_argument("--max-samples", type=positive, default=20000)
    training.add_argument("--debug-steps", type=positive)
    training.add_argument("--report-to", choices=["none", "tensorboard"], default="tensorboard")
    training.add_argument("--dry-run", action="store_true")
    evaluation = sub.add_parser("eval")
    evaluation.add_argument("--run-dir", type=Path, required=True)
    evaluation.add_argument("--result-dir", type=Path)
    evaluation.add_argument("--target-model", type=Path)
    devices = evaluation.add_mutually_exclusive_group()
    devices.add_argument("--gpu", help="one evaluation GPU; default: 0")
    devices.add_argument("--gpus", help="comma-separated GPUs, one dataset process per GPU")
    evaluation.add_argument("--batch-sizes", nargs="+", type=positive, default=[1, 32])
    evaluation.add_argument("--datasets", nargs="+", choices=["gsm8k", *VISION_DATASETS], default=["gsm8k", *VISION_DATASETS])
    evaluation.add_argument("--text-data", type=Path, default=home/"LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/DeepSpec-main-0708-new/eval_datasets/gsm8k.jsonl")
    evaluation.add_argument("--data-dir", type=Path, default=home/"LMM/liyt/datasets/jsonl_datas")
    evaluation.add_argument("--image-root", type=Path, default=home/"LMM/liyt/datasets/jsonl_datas/imgs")
    evaluation.add_argument("--samples", type=positive, default=16)
    evaluation.add_argument("--max-new-tokens", type=positive, default=256)
    evaluation.add_argument("--state-mode", choices=["full_target_replay", "per_token", "compact_replay"], default="per_token")
    evaluation.add_argument("--image-max-pixels", type=positive)
    evaluation.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
    evaluation.add_argument("--loop-sweep", action=argparse.BooleanOptionalAction, default=True,
                            help="evaluate every fixed depth of each loop checkpoint (default: on)")
    evaluation.add_argument("--dry-run", action="store_true")
    reporting = sub.add_parser("report")
    reporting.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if hasattr(args, "run_dir"):
        args.run_dir = args.run_dir.expanduser().resolve()
    if getattr(args, "result_dir", None):
        args.result_dir = args.result_dir.expanduser().resolve()
    if args.action == "train":
        train(args)
    elif args.action == "eval":
        if len(args.datasets) != len(set(args.datasets)):
            parser.error("use unique datasets")
        try:
            ids = evaluation_gpu_ids(args.gpus if args.gpus is not None else args.gpu or "0")
            if args.gpu is not None and len(ids) != 1:
                parser.error("--gpu selects one device; use --gpus for multiple devices")
        except ValueError as error:
            parser.error(str(error))
        evaluate(args)
    else:
        report(args.result_dir)


if __name__ == "__main__":
    main()
