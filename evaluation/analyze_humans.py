"""Score the blinded human ratings and compare them with the automated metrics.

Layout expected (one folder per rater, holding their filled-in copies)::

    <blinded>/blind_key.jsonl
    <raters>/<rater>/pairwise.csv      same_object_choice = A | B | tie
    <raters>/<rater>/scorecard.csv     per-image ratings incl. duplicate_tile_pairs

Outputs ``results/humans/summary.json`` with:

* H1, the protocol's human endpoint: share of pairs where the rater majority
  picked the LoRA grid as showing the same object more consistently (ties = 0.5),
  with a 95% bootstrap CI clustered by prompt.
* Inter-rater agreement: Cohen's kappa (2 raters) or Fleiss' kappa (3+) on the
  forced choice, Krippendorff's alpha (interval) on the 1-5 and 1-4 ratings.
* The fallback check from the protocol: kappa between the human "any duplicate
  tile pair" majority and the automated ``any_same_view``.
* How often each automated metric picks the same grid as the human majority.
* A DreamSim near-duplicate threshold fitted on 20% of labelled tile pairs and
  tested on the remaining 80%.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

CHOICES = ("A", "B", "tie")


# ------------------------------------------------------------------ agreement stats


def cohens_kappa(r1: list[str], r2: list[str]) -> float:
    labels = sorted(set(r1) | set(r2))
    n = len(r1)
    observed = sum(a == b for a, b in zip(r1, r2, strict=True)) / n
    c1, c2 = Counter(r1), Counter(r2)
    expected = sum(c1[label] * c2[label] for label in labels) / n**2
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def fleiss_kappa(ratings: list[list[str]]) -> float:
    """``ratings[item]`` is the list of labels the raters gave that item."""
    labels = sorted({label for item in ratings for label in item})
    counts = np.array([[item.count(label) for label in labels] for item in ratings], float)
    n = counts.sum(axis=1)
    if not np.all(n == n[0]):
        raise ValueError("Fleiss' kappa needs the same number of ratings per item")
    n = n[0]
    p_item = ((counts**2).sum(axis=1) - n) / (n * (n - 1))
    p_label = counts.sum(axis=0) / counts.sum()
    p_bar, p_e = p_item.mean(), (p_label**2).sum()
    return 1.0 if p_e == 1 else float((p_bar - p_e) / (1 - p_e))


def krippendorff_alpha_interval(units: list[list[float]]) -> float:
    """Krippendorff's alpha, interval metric. ``units[i]`` holds the ratings of item i."""
    pairable = [[v for v in unit if v is not None] for unit in units]
    pairable = [unit for unit in pairable if len(unit) >= 2]
    values = [v for unit in pairable for v in unit]
    n = len(values)
    if n < 2:
        return float("nan")
    d_o = (
        sum(
            sum((a - b) ** 2 for a, b in itertools.permutations(unit, 2)) / (len(unit) - 1)
            for unit in pairable
        )
        / n
    )
    d_e = sum((a - b) ** 2 for a, b in itertools.permutations(values, 2)) / (n * (n - 1))
    return 1.0 if d_e == 0 else 1.0 - d_o / d_e


# ----------------------------------------------------------------------- loading


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_tile_pairs(text: str) -> set[tuple[int, int]]:
    """``"12 34"`` -> {(0, 1), (2, 3)}; tiles are numbered 1-4 for raters."""
    pairs = set()
    for token in text.replace(",", " ").split():
        if len(token) != 2 or not set(token) <= set("1234") or token[0] == token[1]:
            raise ValueError(f"Bad tile pair {token!r}; use two digits 1-4 like 12")
        i, j = sorted(int(c) - 1 for c in token)
        pairs.add((i, j))
    return pairs


def load_raters(raters_dir: Path) -> dict[str, dict]:
    raters = {}
    for folder in sorted(p for p in raters_dir.iterdir() if p.is_dir()):
        pairwise = {row["pair_id"]: row for row in read_csv(folder / "pairwise.csv")}
        images = {(r["pair_id"], r["blind_label"]): r for r in read_csv(folder / "scorecard.csv")}
        for pair_id, row in pairwise.items():
            choice = row["same_object_choice"].strip()
            if choice not in CHOICES:
                raise ValueError(f"{folder.name}/{pair_id}: choice must be A, B or tie")
        raters[folder.name] = {"pairwise": pairwise, "images": images}
    if not raters:
        raise ValueError(f"No rater folders in {raters_dir}")
    return raters


# ---------------------------------------------------------------------- analysis


def majority(labels: list[str]) -> str:
    counts = Counter(labels).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return "tie"
    return counts[0][0]


def cluster_bootstrap_share(
    scores: dict[str, float], clusters: dict[str, str], n_boot=10_000, seed=17
) -> dict:
    by_cluster: dict[str, list[float]] = {}
    for item, score in scores.items():
        by_cluster.setdefault(clusters[item], []).append(score)
    groups = list(by_cluster.values())
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        draw = rng.integers(0, len(groups), len(groups))
        boots.append(np.mean([v for k in draw for v in groups[k]]))
    low, high = np.quantile(boots, [0.025, 0.975])
    return {
        "share": float(np.mean(list(scores.values()))),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def fit_threshold(distances: list[float], labels: list[bool]) -> float:
    """Distance threshold (pair is duplicate if distance <= t) that maximises F1."""
    best_t, best_f1 = 0.0, -1.0
    for t in sorted(set(distances)):
        pred = [d <= t for d in distances]
        tp = sum(p and y for p, y in zip(pred, labels, strict=True))
        fp = sum(p and not y for p, y in zip(pred, labels, strict=True))
        fn = sum(y and not p for p, y in zip(pred, labels, strict=True))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


def f1_score(pred: list[bool], labels: list[bool]) -> dict:
    tp = sum(p and y for p, y in zip(pred, labels, strict=True))
    fp = sum(p and not y for p, y in zip(pred, labels, strict=True))
    fn = sum(y and not p for p, y in zip(pred, labels, strict=True))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "n": len(labels)}


def analyse(blinded: Path, raters_dir: Path, metrics_dir: Path) -> dict:
    import pandas as pd

    key = [json.loads(line) for line in (blinded / "blind_key.jsonl").read_text().splitlines()]
    condition = {(k["pair_id"], k["blind_label"]): k["condition"] for k in key}
    prompt_of = {k["pair_id"]: k["prompt_id"] for k in key}
    seed_of = {k["pair_id"]: int(k["seed"]) for k in key}
    lora_label = {pid: lbl for (pid, lbl), cond in condition.items() if cond != "base"}
    raters = load_raters(raters_dir)
    names = list(raters)
    pair_ids = sorted(prompt_of)

    # H1: forced choice, majority across raters.
    choices = {
        pid: [raters[r]["pairwise"][pid]["same_object_choice"].strip() for r in names]
        for pid in pair_ids
    }
    scores = {}
    for pid, votes in choices.items():
        winner = majority(votes)
        scores[pid] = 0.5 if winner == "tie" else float(winner == lora_label[pid])
    h1 = cluster_bootstrap_share(scores, prompt_of)
    per_rater = {
        r: float(
            np.mean(
                [
                    0.5
                    if raters[r]["pairwise"][p]["same_object_choice"].strip() == "tie"
                    else float(
                        raters[r]["pairwise"][p]["same_object_choice"].strip() == lora_label[p]
                    )
                    for p in pair_ids
                ]
            )
        )
        for r in names
    }

    agreement: dict = {"raters": names}
    votes = [choices[pid] for pid in pair_ids]
    if len(names) == 2:
        agreement["choice_cohens_kappa"] = cohens_kappa(
            [v[0] for v in votes], [v[1] for v in votes]
        )
    elif len(names) > 2:
        agreement["choice_fleiss_kappa"] = fleiss_kappa(votes)
    images = sorted({key for r in names for key in raters[r]["images"]})
    for field in ("identity_consistency_1_5", "distinct_views_1_4"):
        units = [[_num(raters[r]["images"][img].get(field)) for r in names] for img in images]
        agreement[f"{field}_krippendorff_alpha"] = krippendorff_alpha_interval(units)

    # Automated metrics for the same grids.
    grids = pd.read_csv(metrics_dir / "grids.csv")
    pairs = pd.read_csv(metrics_dir / "pairs.csv")
    grid_of = {
        (pid, lbl): f"{prompt_of[pid]}/{seed_of[pid]}/{condition[(pid, lbl)]}"
        for pid in pair_ids
        for lbl in ("A", "B")
    }
    g = grids.set_index("grid_id")

    metric_agreement = {}
    for metric, higher_is_better in (
        ("identity_dino", True),
        ("identity_dreamsim", True),
        ("identity_dino_unmasked", True),
    ):
        hits = []
        for pid in pair_ids:
            winner = majority(choices[pid])
            if winner == "tie":
                continue
            a, b = g.loc[grid_of[(pid, "A")], metric], g.loc[grid_of[(pid, "B")], metric]
            picked = "A" if (a > b) == higher_is_better else "B"
            hits.append(picked == winner)
        metric_agreement[metric] = {"agree": float(np.mean(hits)), "n_decided_pairs": len(hits)}

    # Duplicates: human majority per tile pair vs the pose model and DreamSim.
    dup_rows = []
    for (pid, lbl), grid_id in grid_of.items():
        marked = [
            parse_tile_pairs(raters[r]["images"][(pid, lbl)].get("duplicate_tile_pairs", ""))
            for r in names
        ]
        for i, j in itertools.combinations(range(4), 2):
            votes_dup = sum((i, j) in m for m in marked)
            row = pairs[(pairs.grid_id == grid_id) & (pairs.i == i) & (pairs.j == j)].iloc[0]
            dup_rows.append(
                {
                    "grid_id": grid_id,
                    "human_dup": votes_dup * 2 > len(names),
                    "auto_same_view": bool(row["same_view"]),
                    "dreamsim_distance": float(row["dreamsim_distance"]),
                }
            )
    by_grid: dict[str, dict] = {}
    for row in dup_rows:
        entry = by_grid.setdefault(row["grid_id"], {"human": False, "auto": False})
        entry["human"] |= row["human_dup"]
        entry["auto"] |= row["auto_same_view"]
    human_any = ["dup" if v["human"] else "none" for v in by_grid.values()]
    auto_any = ["dup" if v["auto"] else "none" for v in by_grid.values()]
    duplicates = {
        "human_any_duplicate_rate": float(np.mean([v["human"] for v in by_grid.values()])),
        "auto_any_same_view_rate": float(np.mean([v["auto"] for v in by_grid.values()])),
        "kappa_any_duplicate_vs_any_same_view": cohens_kappa(human_any, auto_any),
        "pose_model_tile_pairs": f1_score(
            [r["auto_same_view"] for r in dup_rows], [r["human_dup"] for r in dup_rows]
        ),
    }
    duplicates["protocol_fallback_triggered"] = (
        duplicates["kappa_any_duplicate_vs_any_same_view"] < 0.4
    )

    grid_ids = sorted(by_grid)
    random.Random(17).shuffle(grid_ids)
    calibration = set(grid_ids[: max(1, len(grid_ids) // 5)])
    fit = [r for r in dup_rows if r["grid_id"] in calibration]
    test = [r for r in dup_rows if r["grid_id"] not in calibration]
    threshold = fit_threshold([r["dreamsim_distance"] for r in fit], [r["human_dup"] for r in fit])
    duplicates["dreamsim_threshold"] = {
        "threshold": threshold,
        "fitted_on_grids": len(calibration),
        "test": f1_score(
            [r["dreamsim_distance"] <= threshold for r in test], [r["human_dup"] for r in test]
        ),
    }

    # Human distinct-view counts, the A2 fallback.
    distinct = {}
    for cond in ("base", "lora"):
        vals = []
        for pid in pair_ids:
            lbl = lora_label[pid] if cond == "lora" else ("A" if lora_label[pid] == "B" else "B")
            ratings = [
                _num(raters[r]["images"][(pid, lbl)].get("distinct_views_1_4")) for r in names
            ]
            ratings = [v for v in ratings if v is not None]
            if ratings:
                vals.append(float(np.mean(ratings)))
        distinct[cond] = float(np.mean(vals)) if vals else None

    return {
        "n_pairs": len(pair_ids),
        "lora_preferred": h1,
        "lora_preferred_per_rater": per_rater,
        "agreement": agreement,
        "metric_vs_human_choice": metric_agreement,
        "duplicates": duplicates,
        "human_distinct_views_mean": distinct,
    }


def _num(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--blinded", type=Path, default=Path("evaluation/results/blinded_r1"))
    parser.add_argument("--raters", type=Path, default=Path("results/humans/raters"))
    parser.add_argument("--metrics", type=Path, default=Path("results/metrics/main"))
    parser.add_argument("--output", type=Path, default=Path("results/humans/summary.json"))
    args = parser.parse_args(argv)
    summary = analyse(args.blinded, args.raters, args.metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
