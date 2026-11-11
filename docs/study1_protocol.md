# Study 1 protocol: does the four-view LoRA improve identity without duplicating views?

This file fixes the analysis *before* any metric was computed on a LoRA image.
Everything it depends on (controls, base-model noise, the azimuth sign) comes
from real holdout photos and base-model grids only. The git history shows the
order: this protocol is committed before `results/metrics/*` contains a single
LoRA row. Anything decided later is listed under [Deviations](#deviations).

## The question

A 2x2 grid prompt asks for four views of one object. A model can fail in two
opposite ways. It can draw four different-looking objects (identity failure), or
it can draw the same view four times (diversity failure). The second failure
makes naive identity metrics look *better*. So the LoRA has to beat the base
model on identity **and** must not buy that gain by repeating views.

## What is compared

| Arm | What it is |
| --- | --- |
| `base` | FLUX.1-dev, no adapter |
| `r1` | `output/study1_pilot/study1_pilot.safetensors` (step 500, unrecorded seed) |
| `r2` | `output/study1_pilot_v2/study1_pilot_v2.safetensors` (same config, unrecorded seed) |
| `r3` | `output/study1_pilot_r3/study1_pilot_r3.safetensors` (same config, `training_seed: 43`) |

All three replicates use the config in `configs/study1_pilot.yaml`: rank 16,
alpha 16, 500 steps, lr 1e-4, batch 1 x grad-accum 4, 381 training composites.

Generation: the 42 concise holdout prompts in `evaluation/prompts_holdout.jsonl`,
seeds 1001-1004, 1024x1024, 20 steps, guidance 3.5, FLUX transformer and T5 in
8-bit (bitsandbytes), adapters unfused at scale 1.0, a fresh CPU generator per
image. Base is rendered once and shared by all arms. That gives 42 x 4 x 4 = 672 grids.

The holdout prompts all use one canonical layout: top-left front, top-right right
side, bottom-left back, bottom-right left side. The real photos they were written
from do not follow that layout, so the real composites are scored against their
own Gemini captions, never against these prompts.

### Why 8-bit and not the July bf16 run

The July holdout run (bf16, CPU offload, seeds 1001/1002) re-renders bit-exactly
today. But it averaged about 5 minutes per image on this machine, against about
25 seconds in 8-bit, which fits entirely on the 24 GB GPU. The bf16 grids are
kept as a quantization check (see secondary analyses), not mixed into the
primary comparison.

## Provenance checks done before this protocol

* **Step-500 checkpoint.** `study1_pilot.safetensors` was re-saved by AI Toolkit
  on Jul 19 and Jul 27, so its file hash changed (`4f473e...` to `bc410b...`). The
  training samples rendered after each save are byte-identical. Re-rendering the
  July 8-bit comparison with today's file reproduces the step 100, 300 and 500
  images bit-exactly. Same tensors, new header.
* **The July 8-bit "base" image was stale.** It was produced by a first run that
  crashed before loading any adapter, then marked "reused" by every later run
  while the 8-bit loading code was being fixed. Today it does not reproduce (97%
  of pixels differ), while every LoRA image does. The July side-by-side therefore
  compared the LoRA against a base made by different code. New manifests record
  the git commit with every image, and the analysis flags runs that span commits.

## Metrics

All scores are computed per grid on the four 512x512 quadrants. Each tile is
segmented with rembg (`isnet-general-use`), and identity and pose use the object
crop on a flat grey background.

| Metric | Definition | Why |
| --- | --- | --- |
| `identity_dino` | mean of the 6 pairwise cosines of DINOv2-B/14 CLS embeddings of the masked crops | instance-level features, the usual identity backbone |
| `identity_dreamsim` | 1 - mean pairwise DreamSim distance of the masked crops | tuned on human similarity judgements of diffusion images |
| `identity_dino_unmasked` | as `identity_dino` on whole tiles | sensitivity check only: shows how much a shared background inflates identity |
| `distinct_views` | connected components after merging tile pairs whose camera azimuth differs by < 30 deg (Orient Anything V2, relative pose) | 4 = four different viewpoints, 1 = one view repeated |
| `any_same_view` | at least one tile pair < 30 deg apart | the duplicate rate used in the decision rule |
| `pose_accuracy` | share of the 6 tile pairs whose measured azimuth change is within 45 deg of the change implied by the caption | does the grid follow the requested layout |
| `valid_grid` | every tile has at least 1% foreground | a grid with an empty tile counts as a failure, not a missing value |

Orient Anything V2 runs once on all 12 ordered tile pairs of a grid. Each pair's
relative azimuth is the circular mean of the i-to-j reading and the reversed
j-to-i reading. Their disagreement (`rel_inconsistency`) is kept as a
reliability signal.

Thresholds are frozen in `evaluation/metrics/run_metrics.py::Thresholds` and
copied into every `run_info.json`.

### Azimuth sign

Orient Anything reports relative azimuth in its own rotation convention. The
caption convention is "front to right side = +90". The sign mapping one onto the
other is chosen once, on the 42 real holdout composites against their own
captions, and then frozen. With sign -1, 41.7% of the 252 caption-implied angle
changes are matched within 45 deg, against 29.4% with +1. So the default is
`azimuth_sign = -1`. The calibration itself is weak, which is part of why pose
accuracy is not used for conclusions (next section).

### Metric validation (controls)

`evaluation/controls.py` builds five versions of each real holdout composite:
`real`, `dup4` (top-left tile x4), `dup4_jitter` (x4 with small crop and
brightness jitter), `shuffled` (tiles reversed), and `mixed` (four objects from four
categories). A metric is only used for conclusions if it passes the checks listed
in `controls.CHECKS`, which were written before the controls were scored.

Results on the 42 holdout composites (`results/metrics/controls/control_checks.json`):

| Metric | Check | Pass | dup4 | dup4_jitter | mixed | real | shuffled |
| --- | --- | :-: | --: | --: | --: | --: | --: |
| identity_dino | dup4 > real > mixed | yes | 1.000 | 0.943 | 0.085 | 0.766 | 0.766 |
| identity_dino | shuffled = real | yes | | | | 0.766 | 0.766 |
| identity_dreamsim | dup4 > real > mixed, jitter > real | yes | 1.000 | 0.958 | 0.273 | 0.774 | 0.774 |
| identity_dino_unmasked | dup4 > real > mixed | yes | 1.000 | 0.941 | 0.101 | 0.801 | 0.801 |
| distinct_views | real >= dup4 + 1 | **no** | 1.000 | 1.024 | 3.071 | 1.881 | 1.881 |
| distinct_views | dup4_jitter <= 1.5 | yes | | 1.024 | | | |
| any_same_view | dup4, dup4_jitter >= 0.9 | yes | 1.000 | 1.000 | 0.690 | 0.929 | 0.929 |
| pose_accuracy | real >= shuffled + 0.15, real >= dup4 + 0.15 | **no** | 0.460 | 0.405 | 0.238 | 0.417 | 0.310 |

What the failures mean, after looking at the photos:

* **The real captures span narrow arcs.** Orient Anything puts consecutive real
  frames 14-29 deg apart (medians), about 65 deg first to last. For several objects
  (the backpack, the table-tennis bat) that is visibly right: the four photos are
  close to one view. The Gemini captions claim much wider spreads (about 135 deg
  first to last), and they correlate only weakly with the measured angles (r =
  0.39). So the failed `distinct_views` check was a wrong assumption about the
  ceiling, not proof that the metric is broken. Both duplicate controls are
  detected cleanly. The bigger consequence is about the LoRA: **its training data
  shows nearby views while its captions name distant ones.** *(Later corrected
  with camera poses: Orient Anything compresses angles to about 0.70x, so the
  true arcs are wider than this. See [Deviations](#deviations).)*
* **Orient Anything is unreliable on top-down shots.** For a hammer filmed from
  above while the camera circles it, the model reports at most 65 deg where the
  photos show about 180 deg. Only 18 of 168 real tiles get a unique-front
  symmetry class.
* **Pose accuracy is invalid on this data.** Caption labels come in 45-deg steps and
  the tolerance is 45 deg, so a duplicated tile (relative angle 0) already
  "matches" every caption that names neighbouring views. That is why dup4 beats
  real.

Real tiles are also letterboxed: portrait and landscape frames are padded with
black bars. The worst tile in a real composite is on average 44% bars, while base
FLUX grids have none. `max_letterbox_fraction` is reported for every arm because
a LoRA trained on these composites may learn to draw bars.

## Analysis

Unit of analysis: the prompt. For each prompt, an arm's value is the mean over
its 4 seeds. LoRA-minus-base differences are paired by prompt and seed.

* **Point estimate and 95% CI:** cluster bootstrap over prompts (10,000
  resamples, seed indices resampled within prompt with the same draw for both
  arms), `evaluation/metrics/stats.py::paired_cluster_bootstrap`.
* **Test:** two-sided Wilcoxon signed-rank on the 42 per-prompt differences, with
  Holm correction across the two automated primary endpoints.
* **Pooled LoRA arm:** the mean of r1, r2 and r3 in each prompt-seed cell. Each
  replicate is also reported separately.

### Primary endpoints and decision rule

| ID | Endpoint | Required for "improves identity without duplicating" |
| --- | --- | --- |
| A1 | delta `identity_dino` (pooled LoRA - base) | 95% CI lower bound > 0, and the same sign in all 3 replicates |
| A2 | delta `distinct_views` | 95% CI lower bound > -0.25 (non-inferiority: at most a quarter of a view lost) |
| H1 | human 2AFC: share of pairs where raters pick the LoRA grid as "more clearly the same object" | 95% CI lower bound > 0.5 |
| V | delta `valid_grid` rate | point estimate > -0.05 |

Holm correction covers A1 and A2. `identity_dreamsim` is reported next to A1 as
a robustness check and should agree in sign. `any_same_view`, `pose_accuracy`
and `max_letterbox_fraction` are secondary. If the automated duplicate measure
(`any_same_view`) agrees with the human "any duplicate tile pair" label at
Cohen's kappa < 0.4, A2 is replaced by the human distinct-view count from the
scorecard, since then the pose model cannot be trusted on generated grids either.

These gates follow the original research plan, sized to what a 42-prompt pilot
can resolve. If A1 and H1 disagree, the result is reported as a disagreement between
the metric and people, not as a win for either. If any gate fails, the
conclusion is: "no evidence that this LoRA improves identity without a
diversity cost at this scale." A raw increase in `identity_dino` alone is never
enough.

### What the study can detect

Generation noise alone, measured as base seed 1001 minus base seed 1002 on the
bf16 holdout grids, implies the following minimum detectable effects for 42
prompts x 4 seeds (alpha 0.05 two-sided, 80% power). They are optimistic,
because they ignore prompt-by-LoRA interaction:

| Metric | Base mean (bf16, 84 grids) | SD of single-seed differences | MDE at 42 prompts x 4 seeds |
| --- | --: | --: | --: |
| identity_dino | 0.660 | 0.200 | 0.043 |
| identity_dreamsim | 0.706 | 0.163 | 0.035 |
| any_same_view | 0.821 | 0.468 | 0.101 |
| distinct_views | 2.18 | 1.135 | 0.245 |

For reference, the real holdout composites score 0.766 on `identity_dino`, so the
base model leaves about 0.1 of headroom, a little over twice the MDE.

## Secondary and exploratory analyses (not used for the verdict)

* LoRA scale sweep {0, 0.25, 0.5, 0.75, 1.0, 1.5} on r1, 12 prompts
  (`evaluation/prompts_holdout_sweep.jsonl`, 3 per category group drawn with
  seed 17), seeds 1001-1002. The scale-0 arm must be bit-identical to base. If it
  is not, the adapter plumbing is broken and nothing else is reported.
* Checkpoint sweep: r1 at steps 100, 200, 300, 400, 500 on the same 12 prompts.
* Quantization: the LoRA-minus-base delta on seeds 1001-1002 in bf16 (July run)
  against 8-bit (this run), same prompts.
* Background effect: `identity_dino_unmasked` minus `identity_dino`, by arm.
* Agreement of every automated metric with the human majority, and a DreamSim /
  pHash near-duplicate threshold fitted on 20% of the human duplicate labels and
  tested on the rest.

## Decisions made after seeing controls and base data (before any LoRA data)

Recorded here because they were made after looking at real and base-model
numbers. None of them used a LoRA image.

1. **Duplicate gate.** The research plan's gate was "duplicate-rate increase,
   upper 95% bound below 5 points". At this sample size the MDE for that rate is
   10 points, so the gate would fail whatever the LoRA did. Replaced by the
   `distinct_views` non-inferiority gate (A2), whose margin of -0.25 views matches
   its MDE.
2. **Pose accuracy demoted** to exploratory, because it fails its controls (see
   above).
3. **`max_letterbox_fraction` added** after noticing black bars in the real
   training composites.
4. **Azimuth sign frozen at -1** from the real-composite calibration.
5. A tile below the 1% foreground floor is treated as "no object" by the pose
   model too (it previously crashed the authors' crop helper). No cached control
   tile was below the floor, so no earlier result changed.

## Deviations

Any change after this file was committed is listed here with its date and reason.

### 1. Camera poses show Orient Anything compresses angles (2026-11-11, before any LoRA metric)

MVImgNet ships COLMAP reconstructions for part of the data. 189 of the 423
accepted composites, 42 of the 106 abstentions and 19 of the 42 holdout composites
have one, so their true camera arcs can be measured
(`analysis/real_camera_arcs.py`, `results/real_arcs/`). On the 19 holdout
composites, the two disagree like this:

| | COLMAP | Orient Anything |
| --- | --: | --: |
| median step between consecutive tiles | 38 deg | 24 deg |
| median first-to-last span | 122 deg | 70 deg |
| tile pairs under 30 deg ("same view") | 17% | 39% |
| mean distinct views at 30 deg | 3.21 | 1.84 |

Orient Anything's relative azimuths are about 0.70x the true ones (least-squares
slope and median ratio agree). Its error is small for nearby views (median 13 deg
below 30 deg) and large for distant ones (50 deg above 90 deg). The sign agrees
with COLMAP, and Gemini's labels have the opposite sign, which confirms the frozen
caption-to-pose-model sign of -1.

So the "narrow arcs" paragraph above is only half right. The real views are closer
together than the captions claim (true median span 109 deg over all posed accepted
composites, against 180 deg path length implied by the Gemini labels), but much
less close than Orient Anything said. The failed `distinct_views` control is partly
the metric's compression, not only the data.

What changes:

* **A2 stays as registered** (30 deg threshold on Orient Anything angles). A
  uniform compression shrinks differences between arms but does not bias their
  sign, so the paired comparison is still valid, just less sensitive.
* **Added sensitivity analysis A2-cal:** `distinct_views` recomputed from
  `pairs.csv` with a 21 deg threshold, which is 30 deg x 0.70 and also the
  threshold that best reproduces the COLMAP same-view labels (balanced accuracy
  0.78; 21-24 deg in leave-one-composite-out). It is reported next to A2 and does
  not enter the verdict. Caveat: only 19 truly same-view pairs went into it.
* Absolute `distinct_views` numbers are reported as "Orient Anything views", not
  as view counts.

The main generation run had started but no generated LoRA grid had been scored
when this was written. The git timestamp of this section precedes
`results/metrics/main/`.

### 2. Notes written after the LoRA results (2026-11-12)

Nothing in the registered analysis changed. The verdict in
[`results/study1/report.md`](../results/study1/report.md) is the rule above applied
as written: A1 fails (pooled delta `identity_dino` +0.006, 95% CI -0.009 to
+0.020, though all three replicates are positive), A2 and V pass, H1 is pending
and cannot change the outcome.

**Provenance flag on the main run.** Every main-run record says
`python_sources_dirty: true`. At launch, the only uncommitted Python file was the
untracked `analysis/build_explorer.py`, which generation does not import. It was
committed 11 minutes later (`d3f649f`). `git diff 2a4a886 2338908 -- evaluation src`
is empty, and the sweep, run from the clean commit `2338908`, reproduced all 24
shared base images and all 24 shared r1 images byte for byte.

**Post-hoc analysis: how far does the adapter move each image?**
(`analysis/adapter_effect.py`, `results/adapter_effect/`). Added after seeing that
the deltas were small, so it explains the result but is not evidence for or
against the hypothesis.

* With prompt and seed fixed, the median per-tile DreamSim distance between the
  base and LoRA images is 0.025-0.045, against 0.30 between two seeds of the base
  model. It grows with scale (0.009 at 0.25 to 0.055 at 1.5) and with training
  step (0.005 at step 100 to 0.034 at step 500).
* The three replicates move each tile in the same direction: the median cosine
  between their DreamSim displacements is 0.49-0.73, against 0.003-0.010 when one
  displacement comes from a different prompt. Their weight updates have cosine
  0.02-0.03 ([lora_notes.md](lora_notes.md)).
* Every training composite has black side bars (median 44% of the tile). No
  generated grid has any, in any arm, at any scale up to 1.5, by the registered
  detector or by a looser one. The LoRA did not pick up the most consistent visual
  feature of its training set.

Reading: the adapter's effect is real and consistent across training runs, but
small. At 500 steps and lr 1e-4 it changes the picture far less than a new seed
does. The most likely explanation for the null result is that this recipe is too
weak, not that In-Context LoRA cannot improve identity. Only a stronger training
run that does learn the bars can test that.
