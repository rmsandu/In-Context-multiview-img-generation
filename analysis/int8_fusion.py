"""Would fusing the LoRA into 8-bit FLUX weights keep it? (No.)

Study 1 runs FLUX.1-dev with LLM.int8-style 8-bit linear layers and keeps the
LoRA as a separate bf16 side path. The alternative, merging dW into W and then
quantizing, rounds every weight to its row's int8 grid. This script measures what
survives of dW when you do that, for one module of every role at mid depth:

* ``changed``: share of weights whose int8 code moves at all (fixed row scale);
* ``cos``: cosine between the change in the quantized weight and the true dW;
* ``error_over_update``: ||Q(W + dW) - (W + dW)|| / ||dW||, int8 rounding error
  relative to the whole update.

    python -m analysis.int8_fusion
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.lora_anatomy import ALPHA_OVER_RANK, classify, load_lora

DEFAULT_LORA = Path("output/study1_pilot/study1_pilot.safetensors")


def fusion_stats(w, dw) -> dict:
    import torch

    scale = w.abs().amax(dim=1, keepdim=True) / 127  # per-output-row absmax

    def quantize(x):
        return torch.round(x / scale) * scale

    change = quantize(w + dw) - quantize(w)
    cos = torch.nn.functional.cosine_similarity(change.flatten(), dw.flatten(), dim=0)
    return {
        "rel_norm": float(dw.norm() / w.norm()),
        "median_update_over_step": float((dw.abs() / scale.expand_as(w)).median()),
        "changed": float((change != 0).float().mean()),
        "cos": float(cos),
        "error_over_update": float((quantize(w + dw) - (w + dw)).norm() / dw.norm()),
    }


def main(argv: list[str] | None = None) -> None:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lora", type=Path, default=DEFAULT_LORA)
    parser.add_argument(
        "--output", type=Path, default=Path("results/lora_anatomy/int8_fusion.json")
    )
    args = parser.parse_args(argv)

    lora = load_lora(args.lora)
    picked: dict[tuple[str, str], str] = {}
    for module in sorted(lora):
        info = classify(module)
        middle = 9 if info["stream"] == "double" else 19  # of 19 double / 38 single blocks
        if info["block"] == middle:
            picked.setdefault((info["stream"], info["role"]), module)

    repo = "black-forest-labs/FLUX.1-dev"
    index = hf_hub_download(
        repo, "transformer/diffusion_pytorch_model.safetensors.index.json", local_files_only=True
    )
    weight_map = json.loads(Path(index).read_text())["weight_map"]
    rows = []
    for (stream, role), module in sorted(picked.items()):
        shard = hf_hub_download(
            repo, f"transformer/{weight_map[module + '.weight']}", local_files_only=True
        )
        with safe_open(shard, "pt") as handle:
            w = handle.get_tensor(f"{module}.weight").float()
        a, b = lora[module]
        dw = torch.from_numpy(ALPHA_OVER_RANK * (b @ a)).float()
        rows.append({"module": module, "stream": stream, "role": role, **fusion_stats(w, dw)})
        print(
            f"{module:48s} changed {rows[-1]['changed']:.3f}  cos {rows[-1]['cos']:.2f}  "
            f"error/update {rows[-1]['error_over_update']:.0f}x"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"lora": str(args.lora), "modules": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
