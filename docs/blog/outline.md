# Blog post outline

Working title options:
* "Four views, one object? Measuring a LoRA that draws multi-view grids"
* "My LoRA looked better. Here's how I checked."
* "The duplication trap: evaluating a four-view FLUX LoRA"

Target: 1,800-2,500 words, 4-5 figures, links into the explorer for every claim
about images. The prose is yours. Below is the skeleton: per section, the one
point it makes, the evidence that carries it, and prompts for the parts only you
can write. Numbers marked TBD come from `results/study1/summary.json` once the
runs finish.

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
* Figure idea: per-pair scatter of measured vs true relative angle, Orient
  Anything and Gemini side by side, with the y = x line (TBD, from
  `results/real_arcs/pairs.csv`).
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

**Point:** The pre-registered rule, applied as written. TBD.

**Evidence:**
* Gate table (A1 identity, A2 diversity, H1 humans, V validity) with CIs.
* Per-prompt paired deltas: `results/study1/figures/paired_identity_deltas.png`.
* Replicate spread: does each of r1/r2/r3 agree in sign?
* Human 2AFC result and rater agreement (kappa), plus how often each metric
  agreed with the human majority.
* Scale sweep: identity and views from scale 0 to 1.5 (`scale_sweep.png`). Is
  there a dose-response?
* Letterboxing: does the LoRA learn to draw the black bars its training photos had?

**Your angle:** state it plainly. If the result is null, say what effect size the
study could have detected and didn't.

## 7. What I'd do differently (≈250 words)

**Point:** Concrete, short list. Candidates:
* Choose training frames by measured camera angle, not frame index, so the
  captions' "front, right, back, left" is actually true.
* Define left/right in the captioning prompt.
* Pick a pose model only after checking it against ground truth.
* Freeze A (or try rank 4): the anatomy says it wouldn't hurt.
* More prompts rather than more seeds: the unit of analysis is the prompt.

**Close:** link the explorer, the repo, the protocol and the HF model card.

---

## Figures checklist

| # | Figure | Source | Status |
| --- | --- | --- | --- |
| 1 | Example grid, base vs LoRA | explorer thumbnail | after runs |
| 2 | Identity x views plane | `analysis/study1_report.py` | after metrics |
| 3 | Measured vs true relative angle (Orient Anything, Gemini) | `results/real_arcs/angles.png` | done |
| 4 | Subspace overlap | `results/lora_anatomy/subspace_overlap.png` | done |
| 5 | Per-prompt paired deltas | `analysis/study1_report.py` | after metrics |
| 6 | Scale sweep | `analysis/study1_report.py` | after sweep |

## Things to avoid (so it reads like you)

* No "delve", "crucial", "landscape", "in conclusion".
* No bullet-point-only sections in the post itself. These notes are bullets so
  you can turn them into sentences.
* Keep at least one thing that went wrong in every section.
