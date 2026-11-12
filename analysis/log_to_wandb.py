"""Log generation and metric results to W&B, next to the training runs.

One W&B run per evaluation in the project the training runs use
(``incontext-lora``), with ``job_type="evaluation"`` and ``group`` set to the study:

* **config:** generation settings from the manifest (model, quantization, steps,
  guidance, seeds, prompt file hash, software versions, code commits), each
  adapter arm (LoRA hash, scale, training run name, and the W&B training run where
  that mapping is verified), and the metric thresholds.
* **table ``grids``:** one row per grid with every metric, the seconds it took to
  generate, and a small thumbnail.
* **summary:** per-arm metric means plus the report's numbers (gates, verdict,
  paired deltas, S1-S4 for Study 1b).
* **figures:** the report figures (charts only).

Real MVImgNet photos are never uploaded. Control sets are built from real holdout
photos, so they are logged as scores only, and thumbnails are only made from
images listed in a generation manifest under ``GENERATED_ROOTS`` whose hash
matches the metric table.

    .venv-eval/bin/python -m analysis.log_to_wandb study1 --dry-run
    .venv-eval/bin/python -m analysis.log_to_wandb study1
    .venv-eval/bin/python -m analysis.log_to_wandb study1b --only study1b-runA-main
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = "incontext-lora"
THUMBNAIL = 384
GENERATED_ROOTS = (
    Path("evaluation/results/study1_8bit"),
    Path("evaluation/results/study1b_8bit"),
    Path("evaluation/results/holdout"),
)
# Verified by training seed and config name after syncing (2026-11-12). The July
# runs behind r1 and r2 are not mapped: several July W&B runs share their names.
TRAINING_RUNS = {
    "study1_pilot_r3": "ki1yqnpk",
    "study1b_crop_c1": "ytaryzfx",
    "study1b_crop_c2": "95zt4wc9",
    "study1b_crop_c3": "6e0bjcy3",
}
METRICS = (
    "identity_dino",
    "identity_dreamsim",
    "identity_dino_unmasked",
    "distinct_views",
    "any_same_view",
    "pose_accuracy",
    "valid_grid",
    "max_letterbox_fraction",
    "min_pair_dreamsim_distance",
    "min_pair_phash_distance",
)
GENERATION_FIELDS = (
    "base_model",
    "quantization",
    "dtype",
    "guidance_scale",
    "num_inference_steps",
    "width",
    "height",
    "scheduler_class",
    "prompts_path",
    "prompts_sha256",
    "model_cpu_offload",
    "generator_device",
    "lora_fused",
    "software_versions",
)


@dataclass(frozen=True)
class Evaluation:
    name: str
    study: str
    metrics: Path
    manifest: Path | None = None  # None: not a generation run (controls)
    report: Path | None = None
    report_keys: tuple[str, ...] = ()
    figures: tuple[Path, ...] = ()
    extra: dict[str, Path] = field(default_factory=dict)


EVALUATIONS = (
    Evaluation(
        "study1-main",
        "study1",
        Path("results/metrics/main"),
        Path("evaluation/results/study1_8bit/main/generation_manifest.jsonl"),
        Path("results/study1/summary.json"),
        ("primary", "decision", "code_versions"),
        (
            Path("results/study1/figures/identity_vs_views.png"),
            Path("results/study1/figures/paired_identity_deltas.png"),
        ),
        {"adapter_effect": Path("results/adapter_effect/summary.json")},
    ),
    Evaluation(
        "study1-sweep",
        "study1",
        Path("results/metrics/sweep"),
        Path("evaluation/results/study1_8bit/sweep/generation_manifest.jsonl"),
        Path("results/study1/summary.json"),
        ("sweep",),
        (
            Path("results/study1/figures/scale_sweep.png"),
            Path("results/study1/figures/checkpoint_sweep.png"),
        ),
    ),
    Evaluation(
        "study1-holdout-bf16",
        "study1",
        Path("results/metrics/holdout_bf16"),
        Path("evaluation/results/holdout/generation_manifest.jsonl"),
        Path("results/study1/summary.json"),
        ("quantization",),
    ),
    Evaluation(
        "study1-controls",
        "study1",
        Path("results/metrics/controls"),
        extra={"control_checks": Path("results/metrics/controls/control_checks.json")},
    ),
    Evaluation(
        "study1b-controls-crop",
        "study1b",
        Path("results/metrics/controls_crop"),
        extra={"control_checks": Path("results/metrics/controls_crop/control_checks.json")},
    ),
    Evaluation(
        "study1b-runA-main",
        "study1b",
        Path("results/metrics/study1b_main"),
        Path("evaluation/results/study1b_8bit/main/generation_manifest.jsonl"),
        Path("results/study1b/summary.json"),
        (
            "primary",
            "decision",
            "reading",
            "base_identical",
            "S1_crop_minus_letterbox",
            "S2_adapter_strength",
            "S3",
            "S4_bars",
            "code_versions",
        ),
        (Path("results/study1b/figures/crop_vs_letterbox.png"),),
    ),
    Evaluation(
        "study1b-runB-main",
        "study1b",
        Path("results/metrics/runb_main"),
        Path("evaluation/results/study1b_8bit/runb_main/generation_manifest.jsonl"),
        Path("results/study1b/runb/summary.json"),
        (
            "primary",
            "decision",
            "reading",
            "base_identical",
            "T1_runb_minus_runa",
            "T2_adapter_strength",
            "T4_bars",
            "code_versions",
        ),
        (Path("results/study1b/runb/figures/runb_vs_runa.png"),),
    ),
    Evaluation(
        "study1b-runB-sweep",
        "study1b",
        Path("results/metrics/runb_sweep"),
        Path("evaluation/results/study1b_8bit/runb_sweep/generation_manifest.jsonl"),
        Path("results/study1b/runb/summary.json"),
        ("T3_length",),
    ),
)


# ---------------------------------------------------------------------- inputs


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_generated(path: Path, repo: Path = Path(".")) -> bool:
    """True only for files inside one of the generation output trees."""
    resolved = (repo / path).resolve()
    return any(resolved.is_relative_to((repo / root).resolve()) for root in GENERATED_ROOTS)


def generation_config(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Settings shared by every record, per-arm adapter details, and code commits."""
    if not records:
        raise ValueError("Empty generation manifest")
    first = records[0]
    config: dict[str, Any] = {k: first.get(k) for k in GENERATION_FIELDS if k in first}
    config.setdefault("quantization", None)  # older manifests: bf16, no field
    if config["quantization"] is None:
        config["quantization"] = "none (bf16)"
    for key in GENERATION_FIELDS:
        values = {json.dumps(r.get(key), sort_keys=True) for r in records if key in r}
        if len(values) > 1:
            config[key] = f"varies across records ({len(values)} values)"
    config["seeds"] = sorted({r["seed"] for r in records})
    config["n_prompts"] = len({r["prompt_id"] for r in records})
    config["n_images"] = len(records)
    arms: dict[str, dict[str, Any]] = {}
    for r in records:
        # Base records carry the path of the adapter loaded after them; drop it.
        adapter = r.get("adapter_active", True)
        name = r.get("training_run_name") if adapter else None
        arm = arms.setdefault(
            r["condition"],
            {
                "adapter_active": adapter,
                "lora_path": r.get("lora_path") if adapter else None,
                "lora_sha256": r.get("lora_sha256") if adapter else None,
                "lora_scale": r.get("lora_scale") if adapter else 0.0,
                "training_run_name": name,
                "wandb_training_run": TRAINING_RUNS.get(name or ""),
                "n_images": 0,
                "seconds": [],
            },
        )
        arm["n_images"] += 1
        if r.get("elapsed_seconds"):
            arm["seconds"].append(r["elapsed_seconds"])
    for arm in arms.values():
        seconds = arm.pop("seconds")
        arm["median_seconds_per_image"] = float(np.median(seconds)) if seconds else None
    config["arms"] = arms
    commits = Counter()
    for r in records:
        version = r.get("code_version") or {}
        dirty = " (dirty)" if version.get("python_sources_dirty") else ""
        commits[f"{(version.get('git_commit') or 'unknown')[:12]}{dirty}"] += 1
    config["code_commits"] = dict(commits)
    return config


def grid_rows(
    grids, manifest_records: list[dict[str, Any]] | None, manifest: Path | None = None
) -> list[dict[str, Any]]:
    """Metric rows joined with generation time and prompt group from the manifest.

    Joined on the image path, not the hash: an adapter at scale 0 can render the
    same bytes as the base image.
    """
    by_path = {}
    for r in manifest_records or []:
        by_path[(manifest.parent / r["output_path"]).as_posix()] = r
    rows = []
    for g in grids.to_dict("records"):
        record = by_path.get(Path(g["image_path"]).as_posix())
        if manifest_records is not None:
            if record is None:
                raise ValueError(f"{g['grid_id']} is not in the generation manifest")
            if record["output_sha256"] != g.get("image_sha256"):
                raise ValueError(f"{g['grid_id']}: metric table and manifest hashes differ")
        row = {
            "grid_id": g["grid_id"],
            "prompt_id": g["prompt_id"],
            "prompt_group": (record or {}).get("prompt_group"),
            "seed": int(g["seed"]),
            "arm": g["arm"],
            "seconds": (record or {}).get("elapsed_seconds"),
            "image_path": g["image_path"],
            "image_sha256": g.get("image_sha256"),
        }
        for metric in METRICS:
            value = g.get(metric)
            row[metric] = None if value is None or (isinstance(value, float) and np.isnan(value)) else float(value)
        rows.append(row)
    return rows


def arm_means(grids) -> dict[str, dict[str, float]]:
    frame = grids.copy()
    for column in ("any_same_view", "valid_grid"):
        if column in frame:
            frame[column] = frame[column].astype(float)
    present = [m for m in METRICS if m in frame]
    means = frame.groupby("arm")[present].mean()
    return {
        metric: {arm: float(v) for arm, v in means[metric].items() if not np.isnan(v)}
        for metric in present
    }


def thumbnail(
    path: Path, expected_sha256: str, size: int = THUMBNAIL, repo: Path = Path(".")
) -> bytes:
    """JPEG thumbnail of a generated grid, after checking it is generated and unchanged."""
    if not is_generated(path, repo):
        raise ValueError(f"Refusing to upload {path}: not inside a generation output tree")
    data = (repo / path).read_bytes()
    drop_page_cache([repo / path])
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"{path} does not match the hash in the metric table")
    from PIL import Image

    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=float))


# ---------------------------------------------------------------------- logging


def build(evaluation: Evaluation) -> dict[str, Any]:
    """Everything that would be logged for one evaluation, without contacting W&B."""
    import pandas as pd

    grids = pd.read_csv(evaluation.metrics / "grids.csv")
    run_info = json.loads((evaluation.metrics / "run_info.json").read_text())
    records = read_jsonl(evaluation.manifest) if evaluation.manifest else None
    config: dict[str, Any] = {
        "study": evaluation.study,
        "metrics_dir": evaluation.metrics.as_posix(),
        "metric_version": run_info.get("metric_version"),
        "thresholds": run_info.get("thresholds"),
        "n_grids": len(grids),
        "images_uploaded": records is not None,
    }
    if records is not None:
        config["generation"] = generation_config(records)
    summary: dict[str, Any] = {"mean": arm_means(grids)}
    if evaluation.report and evaluation.report.is_file():
        report = json.loads(evaluation.report.read_text())
        summary["report"] = {k: report[k] for k in evaluation.report_keys if k in report}
    for key, path in evaluation.extra.items():
        if path.is_file():
            value = json.loads(path.read_text())
            if isinstance(value, list):  # control checks: one entry per check
                value = {f"{c['metric']}: {c['check']}": c for c in value}
            summary[key] = value
    return {
        "config": json_safe(config),
        "summary": json_safe(summary),
        "rows": grid_rows(grids, records, evaluation.manifest),
        "figures": [p for p in evaluation.figures if p.is_file()],
    }


def log(evaluation: Evaluation, payload: dict[str, Any], *, entity: str | None) -> str:
    import wandb

    run = wandb.init(
        project=PROJECT,
        entity=entity,
        id=f"eval-{evaluation.name}",
        name=f"eval-{evaluation.name}",
        group=evaluation.study,
        job_type="evaluation",
        config=payload["config"],
        resume="never",
    )
    with_images = payload["config"]["images_uploaded"]
    columns = ["thumbnail"] if with_images else []
    columns += [k for k in payload["rows"][0] if k not in ("image_path", "image_sha256")]
    table = wandb.Table(columns=columns)
    for row in payload["rows"]:
        values = [row[c] for c in columns if c != "thumbnail"]
        if with_images:
            jpeg = thumbnail(Path(row["image_path"]), row["image_sha256"])
            from PIL import Image

            values = [wandb.Image(Image.open(io.BytesIO(jpeg)), file_type="jpg")] + values
        table.add_data(*values)
    run.log({"grids": table})
    for figure in payload["figures"]:
        run.log({f"figures/{figure.stem}": wandb.Image(str(figure))})
    run.summary.update(payload["summary"])
    url = run.url
    run.finish()
    return url


def drop_page_cache(paths) -> None:
    """Hand page cache back to Windows; see the WSL GPU-spill note in the project memory."""
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study", choices=sorted({e.study for e in EVALUATIONS}))
    parser.add_argument("--only", action="append", help="evaluation name (repeatable)")
    parser.add_argument("--entity", help="W&B entity (default: the API key's default)")
    parser.add_argument("--dry-run", action="store_true", help="build and check, upload nothing")
    args = parser.parse_args(argv)

    selected = [
        e for e in EVALUATIONS if e.study == args.study and (not args.only or e.name in args.only)
    ]
    for evaluation in selected:
        if not (evaluation.metrics / "grids.csv").is_file():
            print(f"{evaluation.name}: skipped, {evaluation.metrics} has no grids.csv yet")
            continue
        payload = build(evaluation)
        if payload["config"]["images_uploaded"]:
            for row in payload["rows"]:  # check every image before anything is uploaded
                if not is_generated(Path(row["image_path"])):
                    raise ValueError(f"{row['image_path']} is not a generated image")
        n, figs = len(payload["rows"]), len(payload["figures"])
        if args.dry_run:
            print(f"{evaluation.name}: {n} rows, {figs} figures, images={payload['config']['images_uploaded']}")
            continue
        if args.entity is None:
            import wandb

            args.entity = wandb.Api().default_entity
        try:
            import wandb

            wandb.Api().run(f"{args.entity}/{PROJECT}/eval-{evaluation.name}")
            print(f"{evaluation.name}: already logged, skipping (delete the W&B run to relog)")
            continue
        except (ValueError, wandb.errors.CommError):
            pass
        url = log(evaluation, payload, entity=args.entity)
        print(f"{evaluation.name}: {n} rows, {figs} figures -> {url}")


if __name__ == "__main__":
    main()
