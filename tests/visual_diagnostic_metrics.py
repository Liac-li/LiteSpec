"""Pure-Python metrics: fixed-reference hits are NOT counterfactual suffix verification."""
from collections import defaultdict


def prefix_length(hits):
    n = 0
    for hit in hits:
        if not hit:
            break
        n += 1
    return n


def compare_conditions(teacher, conditions):
    base = conditions["disabled"]["tokens"]
    if not teacher or len(base) != len(teacher):
        raise ValueError("nonempty, equal-length proposal/reference arrays required")
    base_hits = [a == b for a, b in zip(base, teacher)]
    base_prefix = prefix_length(base_hits)
    for row in conditions.values():
        if len(row["tokens"]) != len(teacher):
            raise ValueError("proposal/reference lengths differ")
        hits = [a == b for a, b in zip(row["tokens"], teacher)]
        accepted = prefix_length(hits)
        row.update(
            reference_hits=hits, reference_committed_length=1 + accepted,
            first_candidate_hit=hits[0],
            repairs=sum(h and not b for h, b in zip(hits, base_hits)),
            breaks=sum(b and not h for h, b in zip(hits, base_hits)),
            first_candidate_repaired=hits[0] and not base_hits[0],
            first_candidate_broken=base_hits[0] and not hits[0],
            first_rejection_opportunities=int(base_prefix < len(hits)),
            first_rejection_repaired=base_prefix < len(hits) and hits[base_prefix],
            prefix_improved=accepted > base_prefix,
            prefix_worsened=accepted < base_prefix,
            prefix_delta=accepted-base_prefix,
        )
    return conditions


def position_bucket(offset):
    return "first_round" if offset == 0 else "early_1_31" if offset < 32 else "middle_32_127" if offset < 128 else "late_128_plus"


def summarize_probes(rows):
    """Anchor-weighted rates; denominators and per-position counts are explicit."""
    groups = defaultdict(list)
    for row in rows:
        for condition, values in row["conditions"].items():
            for scope in ("all", "first_round" if row["decode_offset"] == 0 else "later_rounds"):
                groups[condition, scope].append(values)
            bucket = position_bucket(row["decode_offset"])
            if bucket != "first_round":
                groups[condition, bucket].append(values)
    output = []
    for (condition, scope), values in groups.items():
        n = len(values)
        positions = max(len(v["reference_hits"]) for v in values)
        counts = [sum(i < len(v["reference_hits"]) for v in values) for i in range(positions)]
        hits = [sum(i < len(v["reference_hits"]) and v["reference_hits"][i] for v in values) for i in range(positions)]
        opportunities = sum(v["first_rejection_opportunities"] for v in values)
        entry = dict(condition=condition, scope=scope, anchors=n,
            first_candidate_hit_rate=sum(v["first_candidate_hit"] for v in values)/n,
            reference_committed_length=sum(v["reference_committed_length"] for v in values)/n,
            first_rejection_opportunities=opportunities,
            first_rejection_repair_rate=sum(v["first_rejection_repaired"] for v in values)/opportunities if opportunities else None,
            per_position_hits=hits, per_position_counts=counts,
            per_position_hit_rates=[h/c for h,c in zip(hits, counts)])
        for key in ("repairs", "breaks", "first_candidate_repaired", "first_candidate_broken", "prefix_improved", "prefix_worsened", "prefix_delta"):
            entry[key+"_per_anchor"] = sum(v[key] for v in values)/n
        if all("teacher_forced_ce" in v for v in values):
            entry["teacher_forced_ce"] = sum(v["teacher_forced_ce"] for v in values)/n
        output.append(entry)
    return output


def summarize_closed_loop(results):
    rows = [r for result in results for r in result.get("round_diagnostics", [])]
    out = []
    for scope in ("all", "first_round", "later_rounds"):
        selected = [r for r in rows if scope == "all" or (r["round_index"] == 0) == (scope == "first_round")]
        eligible = [r for r in selected if r["first_candidate_hit"] is not None]
        out.append(dict(scope=scope, rounds=len(selected), first_candidate_opportunities=len(eligible),
            first_candidate_hit_rate=sum(r["first_candidate_hit"] for r in eligible)/len(eligible) if eligible else None,
            mean_committed_length=sum(r["committed_length"] for r in selected)/len(selected) if selected else None))
    return out
