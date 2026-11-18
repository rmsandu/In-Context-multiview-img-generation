# Blog post outline

Working title options:
* "Four views, one object? Measuring a LoRA that draws multi-view grids"
* "My LoRA looked better. Here's how I checked."
* "The duplication trap: evaluating a four-view FLUX LoRA"

Target: 1,800-2,500 words, 4-5 figures, links into the explorer for every claim
about images. The prose is yours. Below is the skeleton: per section, the one
point it makes, the evidence that carries it, and prompts for the parts only you
can write. Numbers come from `results/study1/summary.json` and
`results/adapter_effect/summary.json`; only the human ratings are still TBD.

---

## 1. The question (≈200 words)

**Point:** In-Context LoRA claims a tiny LoRA can teach FLUX to draw several
related images in one canvas. The paper shows pictures, not numbers. I wanted
a number, for one specific case: four views of the same object.

**Evidence:** one generated grid (pick one where the LoRA visibly helps, and
link the explorer for counterexamples). The 2x2 layout diagram.

**Your angle:**
* Why this problem? What did you want to use multi-view grids for?
* What did you expect before measuring anything?

## 2. "Looks similar" is not evidence (≈250 words)

**Point:** My first read of base vs LoRA was "they look about the same". Eyeballing
a handful of grids can't separate a real effect from seed noise. That needs
paired comparisons, enough prompts, and a rule written down before looking.

**Twist for the end of the section (or for section 6):** the first read was
right. The LoRA moves each image about an eighth as far as a new seed does. But
there was no way to know that, or to know it was a consistent push and not
noise, without measuring.

**Evidence:**
* Pre-registration commit `e73ec48` predates every LoRA metric (git log screenshot).
* MDE table: with 42 prompts x 4 seeds, the smallest detectable identity change
  is 0.043, and 0.25 distinct views.
* The stale July `base.png` story: a cached image from a crashed run was reused
  and differed in 97% of pixels. Provenance checks caught it.

**Your angle:** what was the moment you stopped trusting your eyes?

## 3. The duplication trap (≈300 words)

**Point:** Mean tile-to-tile similarity, the obvious identity metric, is maximised
by drawing the same view four times. So identity has to be read together with
view diversity, never alone.

**Evidence:**
* Controls table: one tile copied 4x scores identity 1.000, real photos 0.766,
  four different objects 0.085 (masked DINOv2).
* Central figure: `results/study1/figures/identity_vs_views.png`, the plane with
  real, dup4 and mixed references.

**Your angle:** a sentence on why this trap is easy to fall into (it is the metric
everyone reaches for first).

## 4. Testing the rulers before measuring (≈400 words)

**Point:** Every metric went through controls first. Some passed, one failed and
was dropped, and one turned out to be miscalibrated. The camera poses that
MVImgNet ships settled it.

**Evidence:**
* Pose accuracy failed (a duplicated grid beat the real photos, 0.46 vs 0.42), so
  it was demoted to exploratory.
* Orient Anything vs COLMAP on 19 real composites: median span 70 deg vs 122 deg;
  0.70x shrinkage; same-view pairs 39% vs 17%.
* Gemini's labels go the other way: path length 180 deg vs 109 deg true. Two
  instruments, opposite biases. The camera poses are the referee.
* Figure: `results/real_arcs/angles.png`, measured vs true angle for both
  instruments. Gemini's "180 deg apart" covers true angles from 15 to 165 deg.
* Recorded as protocol deviation 1, before any LoRA grid was scored.

**Your angle:**
* How did it feel to find your own claim ("narrow arcs") half wrong?
* What would you check first next time?

## 5. What the LoRA changed in the weights (≈350 words)

**Point:** Three runs, same recipe, almost orthogonal weight updates (cosine
0.02). But that is LoRA's parametrisation, not disagreement: A stays at its random
init and B carries the learning.

**Evidence:**
* `results/lora_anatomy/subspace_overlap.png`.
* adaLN modulation layers take the biggest updates at effective rank ~3.
* The int8 aside: fusing the LoRA into 8-bit weights would mostly erase it
  (cosine 0.10-0.31 with the true update).
* Details in `docs/lora_notes.md`.

**Your angle:** what surprised you? Which LoRA "fact" did you have wrong before?

## 6. The verdict (≈400 words)

**Point:** The pre-registered rule, applied as written: **no evidence that the
LoRA improves identity without a diversity cost.** The effect is real but small,
and the reason is the training recipe, not the method.

**Evidence:**
* Gate table (`results/study1/report.md`): A1 identity fails (pooled +0.006,
  95% CI -0.009 to +0.020, MDE 0.043), A2 diversity passes (+0.05 views), V valid
  grids passes, H1 humans pending but cannot change the outcome.
* All three runs agree in sign (+0.004, +0.007, +0.007) and their CIs all
  include zero. Real photos sit at 0.766, so the headroom was there.
* Figure: `results/study1/figures/identity_vs_views.png`, with the zoom panel
  where the four CIs overlap almost completely.
* Per-prompt deltas (`paired_identity_deltas.png`): a symmetric spread from
  -0.11 to +0.10. Bagged snacks (in the explorer) is a good example: identity
  0.69 to 0.80 because the LoRA drew one view four times. That's the duplication
  trap from section 3.
* Dose-response (`scale_sweep.png`): identity +0.000, +0.007, +0.011, +0.019,
  +0.028 at scales 0.25 to 1.5. The same direction, just not enough of it.
* Why so small (post hoc, `results/adapter_effect/`): with prompt and seed fixed,
  the LoRA moves each tile DreamSim 0.03-0.05; a new seed moves it 0.30. The
  three runs move each tile in the same direction (cosine 0.49-0.73 vs 0.01 for
  unrelated tiles), although their weights are orthogonal (section 5).
* The killer detail: every training composite has black side bars over 44% of
  each tile. Not one of 936 generated grids has any. The LoRA did not learn the
  most obvious thing in its data.
* Human 2AFC result and rater agreement (kappa): TBD once ratings are in.

**Your angle:** state it plainly. What would you have bet before seeing it? Is a
null result with a clear reason more or less useful to you than a small win?

## 7. What I'd do differently (≈250 words)

**Point:** Concrete, short list. Candidates:
* Choose training frames by measured camera angle, not frame index, so the
  captions' "front, right, back, left" is actually true.
* Define left/right in the captioning prompt.
* Pick a pose model only after checking it against ground truth.
* Freeze A (or try rank 4): the anatomy says it wouldn't hurt.
* Start with a positive control: train until the LoRA reproduces something
  trivial in the data (the black bars), then ask the real question.
* More prompts rather than more seeds: the unit of analysis is the prompt.

**Close:** link the explorer, the repo, the protocol and the HF model card.

---

## Figures checklist

| # | Figure | Source | Status |
| --- | --- | --- | --- |
| 1 | Example grid, base vs LoRA | explorer (bagged snacks, seed 1001) | done |
| 2 | Identity x views plane | `results/study1/figures/identity_vs_views.png` | done |
| 3 | Measured vs true relative angle (Orient Anything, Gemini) | `results/real_arcs/angles.png` | done |
| 4 | Subspace overlap | `results/lora_anatomy/subspace_overlap.png` | done |
| 5 | Per-prompt paired deltas | `results/study1/figures/paired_identity_deltas.png` | done |
| 6 | Scale sweep | `results/study1/figures/scale_sweep.png` | done |

## Things to avoid (so it reads like you)

* No "delve", "crucial", "landscape", "in conclusion".
* No bullet-point-only sections in the post itself. These notes are bullets so
  you can turn them into sentences.
* Keep at least one thing that went wrong in every section.
