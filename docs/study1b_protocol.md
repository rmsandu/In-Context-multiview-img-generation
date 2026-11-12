# Study 1b protocol: does the LoRA work once the black bars are gone?

This file fixes the analysis of run A *before* any adapter has been trained on
cropped composites. The git history shows the order: this protocol, the crop code
and the three training configs are committed before `output/study1b_crop_c*`
exists, and `scripts/run_study1b.sh` refuses to train or generate while this file
has uncommitted changes. Anything decided later goes under
[Deviations](#deviations).

Run B (more data, longer training) is sketched at the end. Its rules are fixed in
this file before it starts.

## Why

Study 1 letterboxed every portrait frame into a square tile: on average 44% of
each training tile was black bars ([Study 1 protocol](study1_protocol.md)). The
bars are a preprocessing error, not something the model should learn. Study 1
found no evidence of an identity gain, and the adapter changed each image about an
eighth as much as a new seed does. A plausible reason: almost half of every
training image carried no information about the object, and the rest showed it
small, in a strip.

Run A changes only that. If the bars were what kept the adapter weak, the cropped
adapter should move images further from the base model, and may now pass the
Study 1 identity gate.

## What changes and what stays fixed

| | Study 1 (`r1`-`r3`) | Run A (`c1`-`c3`) |
| --- | --- | --- |
| Objects | 381 train / 42 holdout | same |
| Frames | 4 per object | the same files, checked by SHA-256 |
| Captions | Gemini 3.5 Flash, schema v4 | the same files, byte for byte |
| Tile | frame shrunk to fit 512x512, black bars | object-centred square crop, resized to 512x512 |
| Training | rank 16, alpha 16, 500 steps, lr 1e-4, batch 1 x accum 4, EMA 0.99, 8-bit base | same (`configs/study1b_crop_c*.yaml` differ from `study1_pilot_r3.yaml` only in name, folder and seed) |
| Training seeds | unrecorded, unrecorded, 43 | 41, 42, 43 |
| Generation | 42 holdout prompts x seeds 1001-1004, 1024x1024, 20 steps, guidance 3.5, 8-bit, unfused adapter at 1.0 | same |
| Metrics and thresholds | `evaluation/metrics/run_metrics.py::Thresholds` | same |

`c3` and `r3` share training seed 43, so their configs differ only in the run name
and the dataset folder. That pair is the cleanest single comparison (S3 below).

### The crop rule

Implemented in `src/composite_img.py::object_square_box` and
`src/crop_composites.py`, written before any training:

1. Mask the frame (scaled to a long side of 1024) with rembg `isnet-general-use`,
   the masker the metrics already use, threshold 128.
2. The object is the largest 4-connected foreground region. Below 1% of the frame
   it counts as "not found".
3. The crop is the largest square in the frame (side = the short side, so nothing
   is lost across the width of a portrait frame), slid along the long side so its
   centre matches the centre of the object's bounding box, clamped to the frame.
   With no object found, it is centred on the frame.
4. Resize the square to 512x512 (Lanczos). No padding anywhere.

Before being cropped, each composite was rebuilt with the old letterbox rule and
compared with the composite Study 1 trained on. All 423 match pixel for pixel, so
the frames are the same ones, in the same tile order.

### What the crop does to the data (seen before training)

From `results/study1b/crop/` (`build_info.json`, `crop_manifest.jsonl`; copies of the
build records in the git-ignored `training/study1b_crop/`) and a visual check of the crops.
The build is deterministic: rebuilt from the committed code, all 423 images and crop
records came out identical.

* 1,692 frames: 1,568 portrait, 124 landscape. 7 had no object found and were
  centred.
* The object's bounding box is fully inside the crop in 82% of frames. 316 of 423
  composites have all four views fully inside.
* 55 composites have at least one view with more than 25% of the object box cut.
  They are long or hanging objects: clothing on a stand (13, where the mask
  includes the stand), scarves, rackets, ladders, shovels. A square crop of a 9:16
  frame cannot hold an object taller than the frame is wide.
* **All 423 are kept.** Dropping the cut ones would change the training set, which
  is the one thing run A holds fixed. `cut_fraction` per view is in
  `crop_manifest.jsonl`.
* Captions were written from the letterboxed composites and are reused unchanged.
  Where a crop cuts an object, a caption can name a part that is no longer
  visible. This is accepted for the same reason.

### Controls on the cropped real holdout (seen before training)

The control checks were rerun on the 42 cropped holdout composites
(`results/metrics/controls_crop/`). Every check that passed in Study 1 passes again;
the same two fail for the same reasons (`distinct_views` real >= dup4 + 1, and
`pose_accuracy`).

| Real holdout composites | Letterboxed (Study 1) | Cropped |
| --- | --: | --: |
| `identity_dino` | 0.766 | 0.750 |
| `identity_dreamsim` | 0.774 | 0.776 |
| `max_letterbox_fraction` | 0.438 | 0.000 |
| `valid_grid` | 1.000 | 0.976 (one tile with no object found) |

The cropped real score, 0.750, is the new headroom reference: base FLUX scores
0.651 in Study 1, so about 0.10 of room, still a little over twice the MDE.

## Arms

| Arm | What it is |
| --- | --- |
| `base` | FLUX.1-dev, no adapter, rendered again in this run |
| `c1`, `c2`, `c3` | `output/study1b_crop_c{1,2,3}/study1b_crop_c{1,2,3}.safetensors`, step 500 |
| `r1`, `r2`, `r3` | the Study 1 letterbox adapters, images reused from `evaluation/results/study1_8bit/main` |

**Base reproducibility gate.** All 168 base images of this run must be byte-identical
to Study 1's (`scripts/run_study1b.sh check-base`). If any differs, the crop-vs-letterbox
comparisons (S1-S3) are not reported, because the Study 1 adapter images would no
longer share a base with this run. The primary analysis against this run's own base
stays valid either way.

## Primary analysis: the Study 1 rule, unchanged

Exactly the [Study 1 analysis](study1_protocol.md#analysis), with the pooled crop
arm (mean of `c1`-`c3` per prompt-seed cell) in place of the pooled letterbox arm:
cluster bootstrap over prompts (10,000 resamples), two-sided Wilcoxon, Holm across
A1 and A2, and the A2-cal sensitivity analysis from Study 1's deviation 1.

| ID | Endpoint | Required |
| --- | --- | --- |
| A1 | delta `identity_dino` (pooled crop - base) | 95% CI lower bound > 0, same sign in `c1`, `c2` and `c3` |
| A2 | delta `distinct_views` | 95% CI lower bound > -0.25 |
| V | delta `valid_grid` rate | point estimate > -0.05 |
| H1 | human 2AFC, crop vs base, "more clearly the same object" | 95% CI lower bound > 0.5 |

H1 is only collected if A1, A2 and V pass, with the same rater guide and pack format
as Study 1. If any automated gate fails, the verdict is "no evidence that this LoRA
improves identity without a diversity cost at this scale", and human ratings
cannot change it.

## Key secondary analyses: crop vs letterbox

These are the reason for run A. They are pre-specified but do not enter the
verdict above.

| ID | Comparison | Estimate |
| --- | --- | --- |
| S1 | delta `identity_dino`, pooled crop - pooled letterbox, paired by prompt and seed | cluster bootstrap 95% CI; `identity_dreamsim` alongside |
| S2 | adapter strength: per tile, the DreamSim distance between an adapter's image and the base image with the same prompt and seed (as in `analysis/adapter_effect.py`) | median over tiles for crop and for letterbox; the ratio crop / letterbox with a 95% CI from a bootstrap over prompts |
| S3 | `c3` vs `r3` only (same training seed) | S1 and S2 for that pair |
| S4 | bars in generated grids | `max_letterbox_fraction` and the looser `soft_bar_fraction`, by arm |

How the results will be read, written in advance:

* **A1 passes.** Removing the bars was enough for a measurable identity gain with
  this recipe. H1 then decides the full claim.
* **A1 fails and the S2 ratio's CI includes 1.** The bars were not what kept the
  adapter weak; the recipe (500 steps at lr 1e-4) is. Run B tests that.
* **A1 fails and the S2 ratio's CI lies above 1.** Cropping makes the adapter
  stronger, but not enough to show on identity with 42 prompts.
* **Any crop arm draws bars** (S4 above 0.05 on average): a bug in the crop build;
  stop and investigate.

`results/study1b/report.md` is produced by `analysis/study1b_report.py`, which is
committed before the first crop-arm metric is computed.

## Run B (sketch, fixed here before it starts)

What is already settled:

* **Holdout and prompts unchanged:** the same 42 holdout objects, the same 42
  prompts and 12 sweep prompts, so every run compares on one test set.
* **Data:** the 381 run A training composites exactly as built for run A, plus the
  objects whose frames are missing from the local copy of shards `mvi_32` and
  `mvi_40`: 1,367 objects, each with at least four frames in the original zip files
  (40,396 JPEG frames in total). Frames are selected with the existing rule
  (`src/dataset_builder.py::choose_four_views`) from each object's full JPEG list,
  cropped with the run A crop rule, and captioned with the same Gemini model and
  schema. Abstentions are dropped as before. The instance-level split checks
  confirm that no new object is a holdout instance.
* **Existing objects are not rebuilt.** Three Study 1 objects were built from an
  incomplete local frame list (16-23 of their 31-32 frames), so their four views are
  spaced differently than the rule would choose from the full list. They stay as
  they are, so run B's training set contains run A's unchanged.
* **Not included:** `mvi2_31` (MVImgNet 2). Most of its categories are not in the
  local category list, and mixing releases would be a second change on top of
  "more data".

Still open, to be fixed here before training run B: the number of steps (about
2,000), the learning rate, and the number of replicates. They depend on run A's
S2 result.

## Deviations

None yet.
