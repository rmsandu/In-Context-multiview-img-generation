#!/usr/bin/env python3
"""Gradio demo for a locally trained four-view LoRA checkpoint."""

import argparse
from pathlib import Path

import gradio as gr
import torch
from diffusers import FluxPipeline

DEFAULT_BASE_MODEL = "black-forest-labs/FLUX.1-dev"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora-model", type=Path, required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--share", action="store_true")
    return parser


def create_demo(base_model: str, lora_model: Path) -> gr.Interface:
    if not lora_model.is_file():
        raise FileNotFoundError(f"LoRA checkpoint does not exist: {lora_model}")

    pipe = FluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.load_lora_weights(str(lora_model))
    pipe.fuse_lora()
    pipe.set_progress_bar_config(disable=False)

    def infer(prompt: str, steps: int = 10, guidance: float = 4.0):
        if "[FOUR-VIEWS]" not in prompt:
            prompt = f"[FOUR-VIEWS] {prompt}"
        return pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=512,
            width=512,
        ).images[0]

    return gr.Interface(
        fn=infer,
        inputs=[
            gr.Textbox(
                label="Prompt",
                value="[FOUR-VIEWS] a red desk lamp from multiple views",
                lines=2,
            ),
            gr.Slider(4, 50, value=5, step=1, label="Inference steps"),
            gr.Slider(0, 15, value=4, step=0.5, label="Guidance scale"),
        ],
        outputs=gr.Image(type="pil", label="Result"),
        title="Four-View Grid LoRA",
        allow_flagging="never",
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    create_demo(args.base_model, args.lora_model).launch(
        server_name=args.server_name, share=args.share
    )


if __name__ == "__main__":
    main()
