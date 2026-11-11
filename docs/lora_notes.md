# LoRA notes, with this adapter's numbers

Working notes from re-learning LoRA on a model I had actually trained. Every
number below comes from `results/lora_anatomy/` (`analysis/lora_anatomy.py`,
`analysis/int8_fusion.py`) and can be regenerated on CPU.

## The setup

A linear layer `y = W x` gets a side path:

```
y = W x + (alpha / r) * B (A x)        A: r x d_in,  B: d_out x r
```

* `r = 16`, `alpha = 16`, so `alpha / r = 1` and the update is just `dW = B A`.
  Alpha acts like a learning-rate knob for dW. With Adam the step size of A and
  B doesn't depend on the gradient's scale, so multiplying alpha by k makes dW
  move about k times faster (Hu et al. 2021, section 4.1, which is why
  `alpha = r` is a common default). With plain SGD it would be closer to k^2,
  since the gradient of B is also scaled by k.
* 494 linear layers are adapted: 14 per double-stream block x 19 blocks
  (image and text attention projections, both MLPs, both adaLN modulation
  layers) plus 6 per single-stream block x 38 blocks.
* 85.9M trainable parameters, 0.7% of the 12B transformer. The file is 172 MB
  in bf16.
* Training: AdamW 8-bit, lr 1e-4, batch 1 with 4 gradient-accumulation
  steps, 500 steps, about 5 epochs over 381 composites. The base is quantized
  during training too (AI Toolkit `quantize: true`), and the LoRA stays bf16.
* Init: A random, B = 0. So `dW = 0` at step 0 and the model starts out
  exactly as the base model. That's also why the gradient of A is zero at
  step 0: `dL/dA = B^T (dL/dy) x^T` and B is zero.

## How big is the change?

Per module, `||dW||_F / ||W||_F` has a median of **0.02-0.08%** (by role). The
LoRA moves each weight matrix by well under a tenth of a percent, and that is
enough to change what the model draws.

The update is low rank even within its rank-16 budget. Effective rank
(exp-entropy of the singular values, out of 16) is 11-15 for attention and
MLP layers, but **about 3 for the adaLN modulation layers**. Those layers are
also the ones that moved most (median `||dW||` 0.22-0.25, against 0.05-0.18 for
everything else). They turn the timestep and pooled text embedding into the
per-block shift, scale and gate. A low-rank change there is a cheap global
knob, and it is where this LoRA spent most of its update.

## Same recipe, three runs, three different updates

Three training runs with the same data and config end with **whole-update
cosine 0.02-0.03** between any two of them. Read naively, that says they
learned unrelated things. The subspace overlaps say otherwise:

| Pair | A overlap (input side) | B overlap (output side) | cos(dW) |
| --- | --: | --: | --: |
| r1 step 100 vs final | 0.999 | 0.30 | 0.46 |
| r1 step 400 vs final | 1.000 | 0.93 | 0.95 |
| r1 vs r2 | 0.005 | 0.052 | 0.02 |
| r1 vs r3 | 0.005 | 0.072 | 0.03 |
| r2 vs r3 | 0.005 | 0.055 | 0.02 |
| chance (r / d) | 0.005 | 0.005 | |

(Median over modules. Overlap is the mean squared cosine of the principal
angles between the two 16-dimensional subspaces.)

* **A barely moves.** Its row space at step 100 is the same as at step 500
  (0.999). Across seeds it sits at chance, because each run's A is a different
  random draw that training never leaves.
* **B is where the learning is.** Within a run it keeps rotating (0.30 →
  0.93). Across runs it overlaps 10-14x more than chance: the runs agree on
  *which output directions* to push, but each one reads its input through a
  different random A.
* So the weight-space cosine is close to zero by construction. If both runs
  were approximating the same full-rank target `G` through random input
  projections `P1`, `P2`, the expected cosine would be about `r / d = 0.005`.
  The observed 0.02-0.03 is 4-6x that.

This is the asymmetry described by Zhu et al., "Asymmetry in Low-Rank Adapters
of Foundation Models" (ICML 2024): B does the work and a random, frozen A is
nearly as good as a trained one. Here it shows up without any intervention,
just by looking.

The practical lesson for evaluation: **comparing LoRA weights across seeds
tells you little. Compare their outputs.** That is why Study 1 trains three
replicates and treats run-to-run spread as part of the result.

## Training hadn't finished

`||dW||` summed over all modules grows 0.91, 1.41, 2.00, 2.73, 2.96 at steps
100-500. Cosine to the final update climbs 0.46, 0.59, 0.82, 0.95. The update is
still growing at step 500, though the last 100 steps add less. The checkpoint
sweep in Study 1 shows whether the *images* had settled.

## The scale sweep is a walk along dW

At inference the adapter's output is multiplied by a scale `s`:
`W_eff = W + s * dW`. Scale 0 is the base model (checked byte-identical in
8-bit), 1 is the trained adapter, and 1.5 extrapolates past it. Since `dW` is
a fixed direction, the sweep is a straight line through weight space, which
makes it the cleanest dose-response test available.

## Why the adapter stays unfused (and must, in 8-bit)

Fusing means computing `W + dW` once and dropping the side path. In bf16 that
is equivalent up to rounding. But Study 1 runs the base in 8-bit
(LLM.int8-style, one absmax scale per output row), and fusing there would
mean quantizing `W + dW`.

The update is far smaller than one int8 step: the median `|dW|` entry is
0.2-1.4% of its row's quantization step, depending on the layer. Fusing one module of each role
(`analysis/int8_fusion.py`):

* only **0.5-4%** of weights change their int8 code at all, each by a whole step;
* the change that survives has **cosine 0.10-0.31** with the real `dW`;
* the int8 rounding error is **8-41x** the size of the entire update.

So a fused 8-bit model is mostly a noisier base model. Keeping the LoRA as a
bf16 side path on top of the 8-bit base (what peft does with bitsandbytes
layers) preserves it exactly. The same holds for unfusing: it lets one base
load serve every arm, and lets scale 0 be checked against the base.

## Open questions

* Does a run with a *frozen* random A match these images? The asymmetry
  result predicts yes, at half the trainable parameters.
* Would rank 4 do? Effective ranks of 11-15 suggest the budget is used, but
  the adaLN layers at ~3 suggest the important part may be smaller.
* Training for 1,000 steps: does the update keep growing, and do the images
  change?
