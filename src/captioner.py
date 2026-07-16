import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

MODEL_NAME = "gemini-2.5-flash"


def create_client() -> genai.Client:
    """Create a Gemini client when captioning is requested."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is required for caption generation. "
            "Set it in the environment or in a .env file."
        )
    return genai.Client(api_key=api_key)


def generate_caption_composite_grid(
    composite_img_pil: Image.Image,
    category: str,
    *,
    client: Any | None = None,
) -> str:
    """Generate a joint caption for a 2x2 composite image."""
    prompt = (
        "Generate a single-line joint caption for a 2x2 grid containing four views "
        f"of the same {category}. Describe only visible object appearance and details "
        "at each grid position. Do not claim camera angles, front/back orientation, "
        "elevation, or precise camera poses. Use exactly these position tags: "
        "[FOUR-VIEWS], [TOP-LEFT], [TOP-RIGHT], [BOTTOM-LEFT], and [BOTTOM-RIGHT]. "
        "Example style: [FOUR-VIEWS] Four different views of the same blue handbag; "
        "[TOP-LEFT] blue fabric handbag with gold zippers; [TOP-RIGHT] blue fabric, "
        "short handle and visible side pocket."
    )
    caption_client = client or create_client()
    response = caption_client.models.generate_content(
        model=MODEL_NAME,
        contents=[composite_img_pil, prompt],
        config=types.GenerateContentConfig(max_output_tokens=200, temperature=0.3),
    )
    text = response.text.strip()
    return text if text.endswith(".") else f"{text}."
