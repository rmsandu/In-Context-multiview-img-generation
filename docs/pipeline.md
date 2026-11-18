# Pipeline reference

How the data, captions, training and generation steps work, with the commands to
run each one. The README has the short version; the evaluation design is in
[study1_protocol.md](study1_protocol.md).

## 1. Selecting four views per object

Collect or curate groups containing multiple views of the same object or scene.
[MVImgNet](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet) is one suitable source.
The current dataset build found 529 eligible instances and deterministically selected
four spaced views from each; structured annotation accepted 423 of the resulting
composites. The original small-scale experiments used 126 selected images, while the
current Study 1 pilot uses the larger accepted dataset described above.

The four frames are spaced evenly by frame index along each capture video, not by
measured camera angle. For the 231 composites with COLMAP poses, they cover a
median of about 110 deg of camera motion (`analysis/real_camera_arcs.py`).

## 2. Structured captions with Gemini

`python -m src.dataset_builder` generates captioned 2x2 composites from an
MVImgNet-style directory. Gemini credentials are loaded only when captioning starts.
Set `GOOGLE_API_KEY` in your environment or a local `.env` file:

```bash
export GOOGLE_API_KEY=your-key
python -m src.dataset_builder \
  --objects-dir data/ \
  --category-file mvimgnet_category.txt \
  --output-dir training/composites_4view_grid \
  --abstention-dir training/composites_4view_grid_abstention \
  --cache-dir .gemini_cache \
  --model gemini-3.5-flash \
  --limit 63 \
  --tile-width 512 \
  --tile-height 512
```

Gemini 3.5 Flash (`gemini-3.5-flash`) returns a JSON annotation constrained by the
`MultiviewAnnotation` Pydantic output schema rather than a free-form caption. Each
tile gets an absolute horizontal viewpoint, side, vertical angle, framing, visible
features and a confidence. Python validates the annotation and renders the final
caption deterministically. (One gap found later: the prompt never says whose left
"left" is. See [future_work.md](future_work.md).)

Annotations containing an `indeterminate` viewpoint field are not written into the
LoRA training directory. Their composites and annotations
are retained in the abstention directory for evaluating whether the vision-language
model declines ambiguous orientation judgments. If `--abstention-dir` is omitted, it
defaults to a sibling named `<output-dir>_abstention`.

The cache consists of versioned JSON envelopes. Each entry preserves the exact
raw model text, parsed annotation, validation result, model and prompt versions,
latency, composite-image hash, and hashes of the four source images. Old free-form
`.txt` cache entries are left untouched and are not reused by the structured pipeline.

For hundreds of composites, use the resumable Gemini Batch API workflow instead of
the synchronous builder. Preparation creates keyed JSONL chunks, submission records
each job before continuing, and collection downloads, validates, caches, and renders
the completed dataset:

```bash
python -m src.batch_dataset_builder prepare \
  --objects-dir data \
  --category-file data/mvimgnet_category.txt \
  --output-dir training/composites_4view_grid_all \
  --abstention-dir training/composites_4view_grid_all_abstention \
  --cache-dir .gemini_cache \
  --work-dir .gemini_batch/all_data \
  --model gemini-3.5-flash
python -m src.batch_dataset_builder submit --work-dir .gemini_batch/all_data
python -m src.batch_dataset_builder collect --work-dir .gemini_batch/all_data --wait
```

The commands are resumable. `state.json` records every submitted job ID, successful
responses are stored in the versioned cache, and rerunning `prepare` excludes valid
cached annotations while retrying invalid ones. The July 2026 run used the paths in
the example above and produced the counts reported in
the dataset table in the README.

Each image set needs one caption that describes all four views. Writing these by
hand doesn't scale, so a multimodal model does it. The pipeline defaults to Gemini
3.5 Flash; pass `--model` to use another one. Captioning the already-concatenated
composite works better than sending four separate images, and is cheaper.

## 3. Training the LoRA

To train the model, we will use the In-Context LoRA approach on a diffusion transformer model like FLUX.

LoRA (Low-Rank Adaptation) inserts trainable low-rank weight matrices into the model (typically into attention layers) and freezes the original model weights. This drastically reduces the number of parameters that need updating (and thus memory usage), making it feasible to train on a single high-end RTX 4090 NVIDIA GPU. I trained a LoRA adapter on the FLUX model using the prepared multi-image dataset using the repository [AI-toolkit](https://github.com/ostris/ai-toolkit) and used as inspiration the [In-Context LoRA](https://github.com/ali-vilab/In-Context-LoRA/tree/main) config file.

The repository retains the original `config_4views.yaml` and now includes the
reproducible pilot config at `configs/study1_pilot.yaml`. 

### Launching training

You can remove the [Wandb](https://wandb.ai/) login (I use it for monitoring the loss). You need to login into HuggingFace to accept the FLUX model license usage.

```bash
../ai-toolkit/.venv/bin/wandb login
../ai-toolkit/.venv/bin/huggingface-cli login

WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 \
  ../ai-toolkit/.venv/bin/python \
  ../ai-toolkit/run.py \
  configs/study1_pilot.yaml
```

## 4. Development environment

For CPU-only preprocessing development and tests, install the lightweight project
dependencies rather than the CUDA training stack:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[test]"
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest
```

`requirements.txt` is the CUDA 12.1 environment for generation (`.venv`), with the
exact versions in `requirements-gpu.lock`. `requirements-eval.txt` is the separate
metrics environment (`.venv-eval`, `make eval-env`). Install it
only on a compatible GPU system. To run the local Gradio demo, supply the checkpoint
path explicitly:

```bash
python app.py --lora-model models/4views.safetensors
```

To renumber an existing directory of image/caption pairs:

```bash
python -m src.rename_files training/composites_4view_grid --start 1
```

## 5. The Study 1 pilot: split, training, paired generation

The minimal pilot uses a deterministic, source-instance-level 90/10 split of the
423 accepted pairs. Seed 17 produces 381 training pairs and 42 holdout pairs. The
split command verifies matching PNG/TXT stems and rejects source-instance, composite
image SHA-256, or selected source-image SHA-256 overlap across partitions:

```bash
.venv/bin/python -m src.study1_split
```

This writes `training/study1_pilot/train`, `training/study1_pilot/holdout`, and
`training/study1_pilot/split_manifest.jsonl`. The manifest records every source
instance, destination path, split parameter, composite hash, and selected source
image hashes. Re-running the command is idempotent when the existing files match;
it refuses to overwrite a different file or accept stale extra pairs.

Training uses the neighboring AI Toolkit checkout already expected by this project.
The config name determines the output directory: `study1_pilot` writes to
`output/study1_pilot`. To make a genuinely fresh run, preserve any existing directory
under a different name before launching training. From this repository root, launch
the 500-step pilot explicitly with:

```bash
WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 \
  ../ai-toolkit/.venv/bin/python \
  ../ai-toolkit/run.py \
  configs/study1_pilot.yaml
```

The config trains one rank-16/alpha-16 LoRA and samples two fixed holdout captions
every 50 steps. AI Toolkit copies the effective run configuration to
`output/study1_pilot/config.yaml`; keep that file with the resulting checkpoints.
This repository never launches the GPU command automatically.

After training finishes, generate the controlled base/LoRA pairs with the exact final
checkpoint and its copied run configuration:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m evaluation.generate_pairs \
  --lora output/study1_pilot/study1_pilot.safetensors \
  --training-config output/study1_pilot/config.yaml \
  --output-dir evaluation/results \
  --seeds 1001 1002 \
  --cpu-offload
```

This one command produces 32 grids: eight prompts × seeds 1001 and 1002 × base and
LoRA. It generates every pristine base-FLUX result before loading the LoRA, then
generates the LoRA phase with the same prompt, scheduler, guidance 3.5, 20 steps,
1024×1024 resolution, and freshly reset CPU generator for every output. The adapter
is never fused.

Each pair is saved under
`evaluation/results/images/<prompt_id>/<seed>/{base,lora}.png`. The adjacent
`metadata.json` records the prompt group and text, seed, dimensions, inference
settings, base-model revision, scheduler configuration, training-config and prompt
hashes, exact LoRA path and hash, adapter scale, software versions, device/offload
mode, and output hashes. A flat `evaluation/results/generation_manifest.jsonl`
contains one row per generated grid for downstream metrics. The command refuses to
replace an existing paired run unless `--overwrite` is explicitly supplied. After
an interrupted run, repeat the exact command with `--resume`. Resume mode validates
the manifest records, output hashes, prompts, checkpoint, training config, scheduler,
software versions, and generation settings before skipping an output. It generates
only missing records and rebuilds their adjacent metadata. `--resume` and
`--overwrite` are mutually exclusive.

### Several adapters in one run, in 8-bit (what Study 1 used)

bf16 with CPU offload took about 5 minutes per image on a 24 GB card. 8-bit takes
about 27 s for a base image and 42 s with an adapter attached. Study 1 therefore
runs everything in 8-bit and checks it against the earlier bf16 run (protocol,
"Why 8-bit"). Each `--arm LABEL=PATH[@SCALE]` adds one adapter. The base images
are generated once, then each adapter is loaded, used for every prompt and seed,
and unloaded, so only one is resident at a time. Every arm records its own
checkpoint hash and its sibling `config.yaml`.

```bash
.venv/bin/python -m evaluation.generate_pairs --quantization 8bit --resume \
  --prompts evaluation/prompts_holdout.jsonl --seeds 1001 1002 1003 1004 \
  --output-dir evaluation/results/study1_8bit/main \
  --arm r1=output/study1_pilot/study1_pilot.safetensors \
  --arm r2=output/study1_pilot_v2/study1_pilot_v2.safetensors \
  --arm r3=output/study1_pilot_r3/study1_pilot_r3.safetensors
```

`scripts/run_study1.sh` wraps every GPU stage (third training replicate, main run,
scale and checkpoint sweeps, metrics). Each stage is resumable.

Create randomized A/B image copies, a blank scoring CSV, a pairwise 2AFC sheet, and
a separate condition key with:

```bash
.venv/bin/python -m evaluation.create_blinded_scorecard \
  --generation-manifest evaluation/results/study1_8bit/main/generation_manifest.jsonl \
  --arm r1 --seeds 1001 1002 \
  --output-dir evaluation/results/blinded_r1 \
  --blind-seed 17
```

Raters follow [rater_guide.md](rater_guide.md).

Give raters `scorecard.csv` and the adjacent `images/` directory, but withhold
`blind_key.jsonl` until scoring is complete. Run the focused CPU checks with:

```bash
.venv/bin/python -m pytest tests/test_study1_split.py tests/test_evaluation_pairing.py
```
