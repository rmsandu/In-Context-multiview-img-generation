# Research Plan: Four-View LoRA and Gemini Annotation Quality

## Research questions

1. Does a four-view LoRA improve identity consistency across generated views compared
   with base FLUX.1-dev, without merely duplicating the same view?
2. Which Gemini model produces the most accurate, specific, and non-hallucinated
   structured viewpoint annotations for four-view object composites?

The studies share a reproducible data layer but use separate evaluations. Ambiguous
examples remain labelled `indeterminate` in the Gemini benchmark. They are excluded
from pose-conditioned training, but may be used for appearance-only identity training
without uncertain pose words.

## Study 1: Does LoRA improve identity without duplicating views?

### Training conditions

- Reserve the 100-composite Gemini gold benchmark from all LoRA training.
- Train a **pose-conditioned LoRA** only on determinate, high-confidence annotations.
  Its captions include horizontal view, side, vertical angle, framing, and visible
  features.
- Train an **appearance-only identity LoRA** on valid composites, including examples
  with ambiguous or low-confidence pose annotations. Its captions contain only the
  object summary and visible appearance for each tile. They omit horizontal view,
  side, vertical angle, framing, confidence, and the word `indeterminate`.
- Train three replicates of each LoRA with seeds `17`, `29`, and `43`.
- Hold constant FLUX.1-dev, rank and alpha 16, 500 update steps, batch size 4,
  gradient accumulation 4, learning rate `1e-4`, caption dropout `0.05`, optimizer,
  precision, resolution buckets, and all other training settings.
- Treat differences between the two LoRAs as pipeline-level results because their
  dataset sizes differ. Only each LoRA-versus-base comparison supports a causal claim
  about adding that LoRA pipeline.

### Controlled generation

- Create 30 unseen identity prompts, balanced across 15 categories represented during
  training and 15 held-out categories.
- Give every object three to five distinctive attributes, such as material, color,
  markings, and parts.
- Use 15 appearance-only four-view prompts and 15 explicit pose-conditioned prompts,
  balanced across category exposure.
- Generate seeds `1001` through `1008` for every prompt and checkpoint.
- Evaluate base FLUX once and all six LoRA checkpoints with identical prompts, seeds,
  scheduler, guidance, step count, and 1024×1024 grid dimensions. This produces 1,680
  grids.
- Treat malformed grids as failures rather than silently excluding them. Split valid
  grids into fixed quadrants for tile-level evaluation.

### Human and automated evaluation

- Two blinded annotators rate every grid, with adjudication for disagreements.
- Record identity consistency across all four tiles on a 1–5 scale.
- Record the number of genuinely distinct viewpoints from 1–4.
- Mark near-duplicate tile pairs among the six possible pairs.
- Record four-tile grid validity and prompt-attribute fidelity.
- Compute mean pairwise DINOv2 cosine and DreamSim similarity as supporting identity
  measures.
- Use human duplicate labels as primary truth. Calibrate LPIPS and perceptual-hash
  thresholds on a held-out 20% subset and apply them to the remaining outputs.
- Use hierarchical bootstrap confidence intervals over training seed, prompt, and
  generation seed. Apply Holm correction to the two LoRA-versus-base comparisons.

### Decision rule

Conclude that a LoRA improves identity only when all of the following hold:

- Mean human identity increases by at least 0.3 points and its 95% confidence interval
  excludes zero.
- The upper 95% confidence bound for the increase in grids containing any duplicate
  pair is below five percentage points.
- The lower 95% confidence bound for the change in distinct-view count is above
  `-0.25`.
- Four-tile grid validity does not materially decline.

If neither LoRA satisfies every gate, report that there is no evidence that identity
improved without a diversity cost, even if a raw image-similarity score increased.

## Study 2: Which Gemini model annotates viewpoints best?

### Gold benchmark

- Select 100 composites from the available MVImgNet instances and exclude their hashes
  from every LoRA training export.
- Include 40 clear objects with canonical orientation, 30 canonical objects with
  occlusion, cropping, or difficult angles, and 30 objects or views without a
  meaningful or recoverable front/back orientation.
- Split the benchmark deterministically into 40 calibration and 60 locked test
  composites while preserving the three strata.
- Have two annotators independently label every tile. Adjudicate every disagreement
  and report inter-annotator agreement.
- Gold fields are `object_summary`, `horizontal_view`, `side`, `vertical_angle`,
  controlled `framing`, atomic `visible_features`, and whether abstention is required.
- Keep `indeterminate` as a valid gold label. Count a determinate prediction on a
  gold-ambiguous field as a hallucinated pose.

### Models and inference

Benchmark these four model IDs:

- `gemini-3.5-flash`
- `gemini-3.1-pro-preview`
- `gemini-3.1-flash-lite`
- `gemini-2.5-flash`

Use identical images, prompt, Pydantic schema, temperature 0, and output limit. Retain
each model's default thinking behavior because the benchmark targets the model as it
would be deployed in this pipeline. Randomize and interleave request order.

Cache the exact raw output, parsed output, requested and resolved model versions,
prompt and schema versions, latency, token usage, inference configuration, timestamp,
and composite hash. Run one primary response for every composite and three repeated
responses on a fixed 20-composite subset to measure stability.

### Metrics and model selection

- Report schema-valid response rate.
- Report exact pose-tuple accuracy and per-field macro-F1 for horizontal view, side,
  and vertical angle.
- Report framing macro-F1.
- Report abstention precision, recall, F1, over-abstention on clear views, and false
  determination on ambiguous views.
- Score object-summary and visible-feature precision and recall against adjudicated
  atomic claims.
- Measure specificity as correct visible claims per tile and hallucination rate as
  unsupported claims divided by all generated claims.
- Evaluate confidence with Brier score, expected calibration error, and risk–coverage
  curves.
- Report latency, token consumption, and estimated API cost.

Select the production model with the highest exact pose-tuple accuracy subject to:

- False determination on ambiguous fields no greater than 10%.
- Visible-feature claim precision of at least 95%.
- Schema validity of at least 99%.

Break statistical ties by correct visible claims per tile, then latency and cost. If
no model passes the gates, conclude that none is safe for automatic pose-supervision
generation.

## Dataset policies and interfaces

- Convert `framing` to the controlled enum
  `full_object | partial_object | detail | indeterminate`.
- Add a benchmark command that accepts multiple `--model` values and writes per-model
  JSONL plus aggregate CSV and JSON reports.
- Support three explicit export policies:
  - `benchmark` retains every annotation, including `indeterminate`.
  - `pose` requires every pose field to be determinate and every view confidence to
    meet the calibrated threshold.
  - `identity` renders only object summary and visible features and asserts that the
    final caption contains neither pose labels nor `indeterminate`.
- Calibrate the pose threshold on the 40-composite split. Choose the lowest threshold
  whose one-sided 95% lower confidence bound for exact pose accuracy is at least 90%,
  with at least 50% retained coverage.
- Manually audit a stratified 10% of the final pose export. Raise the threshold or
  require manual correction if precision is below 90%.
- Keep benchmark, pose-training, and identity-training outputs in separate directories
  and manifests. Record source hashes to enforce benchmark exclusion.

## Test and reproducibility requirements

- Test controlled schema fields, abstention behavior, confidence filtering, and both
  caption-rendering policies.
- Assert that training exports contain no benchmark hashes.
- Assert that identity captions contain no pose labels and no `indeterminate` token.
- Test model-specific cache isolation, cache reuse, raw-response preservation, and
  resolved model-version recording.
- Test paired generation manifests, fixed seeds, quadrant extraction, malformed-grid
  accounting, duplicate-threshold calibration, and metric aggregation.
- Save environment and package versions, base and LoRA checkpoint hashes, training
  configs, prompt lists, random seeds, gold-label revisions, and per-example scores.

## Implementation milestones

- [x] Deterministic four-view selection and composite construction.
- [x] Pydantic structured Gemini annotations and abstention-aware caching.
- [x] Accepted and abstention output manifests.
- [ ] Controlled framing enum and versioned schema migration.
- [ ] Gold-label format and annotation instructions.
- [ ] Four-model Gemini benchmark runner and report generator.
- [ ] Confidence calibration and three dataset-export policies.
- [ ] Two LoRA training configurations with three seeds each.
- [ ] Paired base/LoRA generation harness.
- [ ] Automated and human evaluation tooling.
- [ ] Final statistical analysis and research report.

## References

- [Gemini model catalog](https://ai.google.dev/gemini-api/docs/models)
- [Gemini structured outputs](https://ai.google.dev/gemini-api/docs/generate-content/structured-output)
- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- [DreamSim: Learning New Dimensions of Human Visual Similarity](https://papers.neurips.cc/paper_files/paper/2023/hash/9f09f316a3eaf59d9ced5ffaefe97e0f-Abstract-Conference.html)
- [LPIPS perceptual similarity](https://richzhang.github.io/PerceptualSimilarity/)

## Assumptions

- The gold benchmark receives two independent human annotations plus adjudication.
- All four Gemini models remain available when the benchmark runs; the benchmark date
  and resolved versions are recorded.
- Human identity and duplicate judgments are primary evidence. Automated similarity
  metrics are supporting evidence.
- Ambiguous annotations remain available for abstention evaluation but never add
  uncertain pose words to training captions.
