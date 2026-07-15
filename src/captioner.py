import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

MODEL_NAME = "gemini-2.0-flash"
DEFAULT_CACHE_DIR = Path(".gemini_cache")


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


def _caption_one(
    img_pil: Image.Image,
    category: str,
    *,
    client: Any | None = None,
) -> str:
    prompt = (
        f"You are describing a photograph of a single {category} object for "
        "multi-view dataset curation. Focus only on the object described by "
        f"the category name {category}. Respond with a comprehensive detailed "
        f"description of the object using ONLY the name from {category}, including "
        "the viewing angle of the camera, ideally in degrees. EXAMPLE ANSWER if "
        "category is backpack: This photo shows a 0-degree angle front-view shot "
        "of a blue backpack with a front pocket and two zippers."
    )
    caption_client = client or create_client()
    response = caption_client.models.generate_content(
        model=MODEL_NAME,
        contents=[img_pil, prompt],
        config=types.GenerateContentConfig(max_output_tokens=100, temperature=0.4),
    )
    return response.text.strip().rstrip(".")


def caption_four_views(
    view_paths: list[Path],
    category_name: str,
    obj_id: str,
    folder_id: str,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    client: Any | None = None,
) -> tuple[list[str], str]:
    """Caption four views and return their clauses and a joint caption."""
    if len(view_paths) != 4:
        raise ValueError(f"Expected exactly four view paths, received {len(view_paths)}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    unique_prefix = f"{category_name}_{obj_id}_{folder_id}_"
    cache_key = f"{unique_prefix}-" + "-".join(p.stem for p in view_paths) + ".json"
    cache_file = cache_dir / cache_key
    if cache_file.exists():
        clauses = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        caption_client = client or create_client()
        clauses = [
            _caption_one(Image.open(path).convert("RGB"), category_name, client=caption_client)
            for path in view_paths
        ]
        cache_file.write_text(json.dumps(clauses), encoding="utf-8")

    joint = (
        f"[FOUR-VIEWS] This set of four images shows different viewing angles of "
        f"the same {category_name}; "
        + "; ".join(
            f"[IMAGE{i + 1}] "
            f"{clause.replace('Here is a description of the object in the image:', '').strip()}"
            for i, clause in enumerate(clauses)
        )
        + "."
    ).replace("\n", "")
    return clauses, " ".join(joint.split())


def generate_caption_composite_grid(
    composite_img_pil: Image.Image,
    category: str,
    *,
    client: Any | None = None,
) -> str:
    """Generate a joint caption for a 2x2 composite image."""
    prompt = (
        "You are an expert data annotator for 3D computer vision. "
        "Generate a single-line joint caption for a 2x2 grid of images showing "
        f"different views of the same {category}. Describe the object and the "
        "camera viewpoint at each grid position. Use exactly these position tags: "
        "[FOUR-VIEWS], [TOP-LEFT], [TOP-RIGHT], [BOTTOM-LEFT], and [BOTTOM-RIGHT]."
    )
    caption_client = client or create_client()
    response = caption_client.models.generate_content(
        model=MODEL_NAME,
        contents=[composite_img_pil, prompt],
        config=types.GenerateContentConfig(max_output_tokens=200, temperature=0.3),
    )
    text = response.text.strip()
    return text if text.endswith(".") else f"{text}."
