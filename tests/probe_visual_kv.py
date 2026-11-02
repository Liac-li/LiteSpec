"""Untimed fixed-reference diagnostics. Counterfactual proposals NEVER enter Target cache."""

import argparse
from dataclasses import replace
import json
import random
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litespec.vlm import train_aligned as aligned
from litespec.vlm.baselines import load_draft
from litespec.vlm.runtime import load_target, prepare_example, read_records
from litespec.vlm.static_batch import generate_static_batch
from litespec.vlm.benchmark import RequestPositionState
from litespec.modeling.draft.visual_kv import gather_evidence


def append_reference(example, tokens):
    inputs = dict(example.inputs)
    ids = inputs["input_ids"]
    tail = ids.new_tensor([tokens])
    inputs["input_ids"] = torch.cat([ids, tail], 1)
    if "attention_mask" in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    if "mm_token_type_ids" in inputs:
        inputs["mm_token_type_ids"] = torch.cat(
            [inputs["mm_token_type_ids"], torch.zeros_like(tail)], 1
        )
    return inputs


from visual_diagnostic_metrics import compare_conditions as compare, summarize_probes


def scoped_evidence(evidence, history_scope):
    if history_scope == "prefill":
        return replace(evidence, valid=evidence.valid & (
            evidence.positions < evidence.decode_start[:, None]))
    return evidence


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-model", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--reference-file", type=Path, help="AR JSONL from same frozen snapshot")
    p.add_argument("--history-scope", choices=["full", "prefill"], default="full")
    p.add_argument("--context-schedules", nargs="+", choices=["always", "first_round", "off"], default=["always", "first_round", "off"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--image-max-tokens", type=int, default=1024)
    args = p.parse_args()
    if min(args.samples, args.stride, args.max_new_tokens, args.image_max_tokens) < 1:
        p.error("positive limits required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    target, processor = load_target(
        args.target_model, device=args.device, dtype=dtype, attention="sdpa"
    )
    from litespec.vlm.memory import configure_image_tokens

    configure_image_tokens(processor, target.config, args.image_max_tokens)
    draft = (
        load_draft(args.checkpoint, attn_implementation="sdpa")
        .to(args.device, dtype)
        .eval()
    )
    if getattr(draft, "visual_kv", None) is None:
        p.error("visual-KV experiment checkpoint required")
    cfg = aligned.load_config(
        Path(__file__).resolve().parents[1]
        / "scripts/config/vlm/sft_student_aligned.yaml"
    )
    cfg.update(num_anchors=1, block_size=draft.block_size)
    online = aligned.make_online(target, draft, cfg)
    mode = draft.config.dlite_config["context_mode"]
    if args.history_scope == "prefill" and mode != "pivot_history":
        p.error("prefill scope requires a trained pivot_history checkpoint")
    records = list(read_records(args.data, args.samples))
    reference_rows = None
    if args.reference_file:
        reference_rows = [json.loads(line) for line in args.reference_file.read_text().splitlines() if line.strip()]
        if [r["sample_id"] for r in reference_rows] != [r["id"] for r in records]:
            raise ValueError("AR reference IDs/order differ from snapshot")
    def get_reference(index, example):
        if reference_rows is not None:
            return reference_rows[index]["result"]["token_ids"][:args.max_new_tokens]
        with RequestPositionState(target).activate():
            return generate_static_batch(target, None, example,
                max_new_tokens=args.max_new_tokens)["token_ids"]
    rng = random.Random(args.seed)
    donors = {}
    for record in records:
        choices = [r for r in records if r.get("image") and r["image"] != record.get("image")]
        if record.get("image") and choices:
            donors[record["id"]] = rng.choice(choices)
    rows = []
    with (args.output_dir / "anchors.jsonl").open("w") as log:
        for index, record in enumerate(records):
            example = prepare_example(
                processor,
                target.config,
                record,
                device=args.device,
                training=False,
                base_dir=args.data.parent,
            )
            reference = get_reference(index, example)
            if len(reference) < draft.block_size:
                continue
            inputs = append_reference(example, reference)
            start = example.prefill_length
            positions = torch.arange(
                start,
                inputs["input_ids"].size(1) - draft.block_size + 1,
                args.stride,
                device=args.device,
            )
            mask = torch.zeros_like(inputs["input_ids"], dtype=torch.float)
            mask[:, start:] = 1
            with RequestPositionState(target).activate():
                output = target(
                    **inputs,
                    use_cache=False,
                    output_hidden_states=True,
                    logits_to_keep=1,
                )
            prompt = torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
            prompt[:, :start] = True
            evidence = gather_evidence(
                output.hidden_states, draft, inputs["input_ids"], prompt, target.config
            )
            evidence = scoped_evidence(evidence, args.history_scope)
            donor_info = None
            wrong = None
            options = [
                r
                for r in records
                if r.get("image") and r.get("image") != record.get("image")
            ]
            if (
                record.get("image")
                and options
                and draft.config.dlite_config["context_mode"]
                not in {"pivot", "pivot_history"}
            ):
                donor = donors[record["id"]]
                donor_example = prepare_example(
                    processor,
                    target.config,
                    {**record, "image": donor["image"]},
                    device=args.device,
                    training=False,
                    base_dir=args.data.parent,
                )
                donor_inputs = append_reference(donor_example, reference)
                with RequestPositionState(target).activate():
                    donor_output = target(
                        **donor_inputs,
                        use_cache=False,
                        output_hidden_states=True,
                        logits_to_keep=1,
                    )
                dp = torch.zeros_like(donor_inputs["input_ids"], dtype=torch.bool)
                dp[:, : donor_example.prefill_length] = True
                wrong = gather_evidence(
                    donor_output.hidden_states,
                    draft,
                    donor_inputs["input_ids"],
                    dp,
                    target.config,
                )
                donor_info = dict(
                    id=donor["id"], prompt_tokens=donor_example.prefill_length
                )
            for anchor in positions.tolist():
                anchors = inputs["input_ids"].new_tensor([[anchor]])
                batch = online.prepare_batch(
                    inputs["input_ids"],
                    output.hidden_states,
                    mask,
                    anchor_positions=anchors,
                    block_keep_mask=torch.ones_like(anchors, dtype=torch.bool),
                )
                contexts = {
                    "correct": (
                        None
                        if draft.config.dlite_config["context_mode"] == "pivot"
                        else evidence
                    ),
                    "disabled": None,
                }
                if wrong is not None:
                    # Donor visual positions are already known at prefill. Rebase
                    # causal metadata only; no RoPE is applied to visual evidence.
                    contexts["wrong_image"] = replace(
                        wrong,
                        positions=torch.zeros_like(wrong.positions),
                        state_prefix=evidence.state_prefix,
                        decode_start=evidence.decode_start,
                    )
                if draft.visual_kv.mode == "visual_recurrent":
                    contexts["reset_state"] = replace(evidence, reset_state=True)
                    if wrong is not None:
                        da = anchors - start + donor_example.prefill_length
                        state = draft.visual_kv.states_at_anchors(wrong, da)
                        contexts["wrong_state"] = replace(
                            evidence, state_override=state
                        )
                for schedule in args.context_schedules:
                    contexts["schedule_"+schedule] = (
                        contexts["correct"] if schedule == "always" or
                        (schedule == "first_round" and anchor == start) else None)
                conditions = {}
                for condition, bank in contexts.items():
                    batch.context_evidence = bank
                    z = online.forward_backbone(
                        batch, seq_len=inputs["input_ids"].size(1)
                    )
                    ids, logits = draft.sample_draft_tokens(
                        draft_hidden=z[:, 0],
                        first_prev_token_ids=inputs["input_ids"][:, anchor],
                        temperature=0,
                    )
                    latent = draft.sequential_head.forward_teacher_forcing(
                        hidden_states=z, prev_token_ids=batch.prev_token_ids
                    )
                    tf = draft.sequential_head.project_logits(latent).float()
                    ce = torch.nn.functional.cross_entropy(
                        tf.reshape(-1, tf.size(-1)), batch.labels.reshape(-1)
                    ).item()
                    conditions[condition] = dict(
                        tokens=ids[0].tolist(), teacher_forced_ce=ce
                    )
                teacher = batch.labels[0, 0].tolist()
                row = dict(
                    id=record["id"],
                    anchor=anchor,
                    decode_offset=anchor - start,
                    history_scope=args.history_scope,
                    donor=donor_info,
                    reference_tokens=teacher,
                    conditions=compare(teacher, conditions),
                )
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
    summary = summarize_probes(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "settings.json").write_text(
        json.dumps(vars(args), default=str, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

