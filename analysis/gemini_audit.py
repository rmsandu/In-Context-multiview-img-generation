"""Study 2 lite: hand-label 30 real composites and grade Gemini's view labels.

    python -m analysis.gemini_audit prepare   # sample 15 accepted + 15 abstention composites
    # fill results/gemini_audit/label_sheet.csv while looking at images/<stem>.jpg
    python -m analysis.gemini_audit score

Composites are sampled from the ones with COLMAP poses (see
analysis/real_camera_arcs.py), at most one per category per stratum, so the
human labels can also be checked against the true camera angles. The label sheet
does not show Gemini's answer. The images are MVImgNet frames and stay local
(gitignored).

The label vocabulary is Gemini's own schema (src/captioner.py), so agreement is
exact string agreement, plus a looser "within 45 deg" on the implied azimuth.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from analysis.real_camera_arcs import annotation_azimuths
from evaluation.metrics.views import TILE_PAIRS, angular_distance, wrap_degrees

TILES = ("top_left", "top_right", "bottom_left", "bottom_right")
PER_STRATUM = 15
SEED = 17
OUT = Path("results/gemini_audit")
SHEET_FIELDS = ("stem", "tile", "horizontal_view", "side", "notes")
VOCABULARY = {
    "horizontal_view": (
        "front",
        "front_three_quarter",
        "side",
        "rear_three_quarter",
        "back",
        "top",
        "bottom",
        "indeterminate",
    ),
    "side": ("left", "right", "neither", "indeterminate"),
}


def load_records() -> dict[str, dict]:
    records = {}
    for source, manifest in (
        ("accepted", Path("training/composites_4view_grid_all/manifest.jsonl")),
        (
            "abstention",
            Path("training/composites_4view_grid_all_abstention/abstention_manifest.jsonl"),
        ),
    ):
        for line in manifest.read_text().splitlines():
            record = json.loads(line)
            record["source"] = source
            record["path"] = manifest.parent / record["output_image"]
            records[Path(record["output_image"]).stem] = record
    return records


def sample(records: dict[str, dict], posed: set[str]) -> list[str]:
    rng = random.Random(SEED)
    chosen = []
    for source in ("accepted", "abstention"):
        pool = sorted(s for s, r in records.items() if r["source"] == source and s in posed)
        rng.shuffle(pool)
        seen: set[str] = set()
        picked = []
        for stem in pool:  # one per category first, then fill up
            category = stem.rsplit("_", 2)[0]
            if category not in seen:
                picked.append(stem)
                seen.add(category)
            if len(picked) == PER_STRATUM:
                break
        picked += [s for s in pool if s not in picked][: PER_STRATUM - len(picked)]
        chosen += picked
    return chosen


def prepare(args) -> None:
    from PIL import Image

    records = load_records()
    posed = {
        row["stem"] for row in csv.DictReader((args.arcs / "composites.csv").open(encoding="utf-8"))
    }
    stems = sample(records, posed)
    rng = random.Random(SEED + 1)
    rng.shuffle(stems)  # accepted and abstention interleaved, so the stratum is not obvious
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    sheet = OUT / "label_sheet.csv"
    if sheet.exists() and not args.force:
        raise SystemExit(f"{sheet} exists; pass --force to overwrite your labels")
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_FIELDS)
        writer.writeheader()
        for stem in stems:
            image = Image.open(records[stem]["path"]).convert("RGB")
            image.thumbnail((1024, 1024))
            image.save(OUT / "images" / f"{stem}.jpg", quality=88)
            for tile in TILES:
                writer.writerow({"stem": stem, "tile": tile})
    (OUT / "sample.json").write_text(
        json.dumps({"seed": SEED, "stems": {s: records[s]["source"] for s in stems}}, indent=2)
    )
    print(f"{len(stems)} composites -> {sheet} (images in {OUT / 'images'})")
    print("Allowed values:", json.dumps(VOCABULARY))


def _azimuth(view: dict) -> float | None:
    return annotation_azimuths({"views": [view]})[0]


def score(args) -> None:
    records = load_records()
    sample_info = json.loads((OUT / "sample.json").read_text())["stems"]
    human: dict[str, dict[str, dict]] = {}
    for row in csv.DictReader((OUT / "label_sheet.csv").open(encoding="utf-8")):
        for field, allowed in VOCABULARY.items():
            if row[field].strip() not in allowed:
                raise SystemExit(
                    f"{row['stem']} {row['tile']}: {field}={row[field]!r} not in {allowed}"
                )
        human.setdefault(row["stem"], {})[row["tile"]] = {
            k: row[k].strip() for k in ("horizontal_view", "side")
        }

    tiles = []
    for stem, labels in human.items():
        gemini = {v["tile"]: v for v in records[stem]["annotation"]["views"]}
        for tile in TILES:
            h, g = labels[tile], gemini[tile]
            h_az, g_az = _azimuth(h), _azimuth(g)
            tiles.append(
                {
                    "stem": stem,
                    "source": sample_info[stem],
                    "tile": tile,
                    "exact": h["horizontal_view"] == g["horizontal_view"]
                    and h["side"] == g["side"],
                    # The Gemini prompt never says whose left "left" is, so also
                    # compare the view family alone (front / three-quarter / side / ...).
                    "view_only": h["horizontal_view"] == g["horizontal_view"],
                    "within45": None
                    if h_az is None or g_az is None
                    else angular_distance(h_az, g_az) <= 45,
                    "human_abstains": h["horizontal_view"] == "indeterminate",
                    "gemini_abstains": g["horizontal_view"] == "indeterminate",
                    "human_az": h_az,
                    "gemini_az": g_az,
                }
            )

    def share(rows, key):
        values = [r[key] for r in rows if r[key] is not None]
        return {"n": len(values), "rate": round(sum(values) / len(values), 3) if values else None}

    both = [t for t in tiles if not t["human_abstains"] and not t["gemini_abstains"]]
    gem_abs = [t for t in tiles if t["gemini_abstains"]]
    hum_abs = [t for t in tiles if t["human_abstains"]]
    summary = {
        "n_composites": len(human),
        "n_tiles": len(tiles),
        "label_agreement_when_both_commit": {
            "exact": share(both, "exact"),
            "view_only": share(both, "view_only"),
            "within_45deg": share(both, "within45"),
        },
        # Abstention as a detector of "a human can't tell either".
        "abstention": {
            "gemini_abstains": len(gem_abs),
            "human_abstains": len(hum_abs),
            "precision": round(sum(t["human_abstains"] for t in gem_abs) / len(gem_abs), 3)
            if gem_abs
            else None,
            "recall": round(sum(t["gemini_abstains"] for t in hum_abs) / len(hum_abs), 3)
            if hum_abs
            else None,
            "false_determination_rate": round(
                sum(not t["gemini_abstains"] for t in hum_abs) / len(hum_abs), 3
            )
            if hum_abs
            else None,
        },
        "by_stratum": {
            source: share([t for t in both if t["source"] == source], "within45")
            for source in ("accepted", "abstention")
        },
    }

    # Relative angles against COLMAP, for human and Gemini labels alike.
    truth = {}
    for row in csv.DictReader((args.arcs / "pairs.csv").open(encoding="utf-8")):
        truth[(row["stem"], int(row["i"]), int(row["j"]))] = float(row["gt_rel_azimuth"])
    by_stem = {}
    for t in tiles:
        by_stem.setdefault(t["stem"], []).append(t)
    errors = {"human": [], "gemini": []}
    for stem, rows in by_stem.items():
        for i, j in TILE_PAIRS:
            if (stem, i, j) not in truth:
                continue
            for who in errors:
                a, b = rows[i][f"{who}_az"], rows[j][f"{who}_az"]
                if a is not None and b is not None:
                    errors[who].append((wrap_degrees(b - a), truth[(stem, i, j)]))
    summary["relative_azimuth_vs_colmap"] = {}
    for who, pairs in errors.items():
        if not pairs:
            continue
        best = min(
            (1, -1),
            key=lambda s: sum(angular_distance(s * m, r) for m, r in pairs),
        )
        errs = sorted(angular_distance(best * m, r) for m, r in pairs)
        other = sorted(angular_distance(-best * m, r) for m, r in pairs)
        summary["relative_azimuth_vs_colmap"][who] = {
            "n_pairs": len(errs),
            "sign": best,
            "median_abs_error_deg": round(errs[len(errs) // 2], 1),
            "within_45deg": round(sum(e <= 45 for e in errs) / len(errs), 3),
            "median_abs_error_other_sign": round(other[len(other) // 2], 1),
        }

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "tiles.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tiles[0]))
        writer.writeheader()
        writer.writerows(tiles)
    print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("prepare", "score"))
    parser.add_argument("--arcs", type=Path, default=Path("results/real_arcs"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    {"prepare": prepare, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
