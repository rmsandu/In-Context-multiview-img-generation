import hashlib
import io
import json
import os
import time
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

MODEL_NAME = "gemini-3.5-flash"
PROMPT_VERSION = "structured-viewpoints-v3"

TILES = ("top_left", "top_right", "bottom_left", "bottom_right")
HORIZONTAL_VIEWS = (
    "front",
    "front_three_quarter",
    "side",
    "rear_three_quarter",
    "back",
    "top",
    "bottom",
    "indeterminate",
)
SIDES = ("left", "right", "neither", "indeterminate")
VERTICAL_ANGLES = (
    "overhead",
    "high_angle",
    "eye_level",
    "low_angle",
    "underside",
    "indeterminate",
)
Tile = Literal["top_left", "top_right", "bottom_left", "bottom_right"]
HorizontalView = Literal[
    "front",
    "front_three_quarter",
    "side",
    "rear_three_quarter",
    "back",
    "top",
    "bottom",
    "indeterminate",
]
Side = Literal["left", "right", "neither", "indeterminate"]
VerticalAngle = Literal[
    "overhead", "high_angle", "eye_level", "low_angle", "underside", "indeterminate"
]
NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ViewAnnotation(BaseModel):
    """Schema for one static tile in the four-view grid."""

    model_config = ConfigDict(extra="forbid")

    tile: Tile
    horizontal_view: HorizontalView
    side: Side
    vertical_angle: VerticalAngle
    framing: NonEmptyString
    visible_features: Annotated[list[NonEmptyString], Field(min_length=1)]
    confidence: Annotated[float, Field(strict=True, ge=0, le=1)]

    @model_validator(mode="after")
    def validate_side_for_viewpoint(self) -> "ViewAnnotation":
        if self.horizontal_view in {
            "front_three_quarter",
            "side",
            "rear_three_quarter",
        } and self.side not in {"left", "right"}:
            raise ValueError(f"side must be left or right for {self.horizontal_view}")
        if (
            self.horizontal_view in {"front", "back", "top", "bottom"}
            and self.side != "neither"
        ):
            raise ValueError(f"side must be neither for {self.horizontal_view}")
        if self.horizontal_view == "indeterminate" and self.side != "indeterminate":
            raise ValueError(
                "side must be indeterminate when horizontal_view is indeterminate"
            )
        return self


class MultiviewAnnotation(BaseModel):
    """Gemini output schema for a complete four-view composite."""

    model_config = ConfigDict(extra="forbid")

    object_summary: NonEmptyString
    views: Annotated[list[ViewAnnotation], Field(min_length=4, max_length=4)]

    @model_validator(mode="after")
    def validate_tiles(self) -> "MultiviewAnnotation":
        tiles = [view.tile for view in self.views]
        if len(tiles) != len(set(tiles)):
            raise ValueError("view tile values must be unique")
        if set(tiles) != set(TILES):
            raise ValueError("views must contain each required tile exactly once")
        if tiles != list(TILES):
            raise ValueError("views must use canonical tile order")
        return self


ANNOTATION_JSON_SCHEMA = MultiviewAnnotation.model_json_schema()

PROMPT_TEMPLATE = """Analyze this 2x2 grid of four views of the same {category}.
Return only JSON matching the supplied schema. Classify every tile independently with
an absolute, static viewpoint. Never use a relative description such as "another side"
or infer a rotation from neighboring tiles.
Return views in this exact order: top_left, top_right, bottom_left, bottom_right.

Use front/back only when the object has a visually meaningful canonical front and back.
If it does not (for example, some can openers), or the orientation is genuinely unclear,
use horizontal_view "indeterminate" and side "indeterminate" rather than guessing.
For side and three-quarter views, side must be left or right. For direct front, back,
top, or bottom views, side must be neither. Describe only visible features.
Use framing for a concise description of how much of the object is visible in the tile,
such as "full object", "cropped object", or "close-up".

Confidence is from 0 to 1.
"""


@dataclass(frozen=True)
class CaptionResponse:
    annotation: dict[str, Any] | None
    raw_response_text: str
    model_id: str
    prompt_version: str
    latency_ms: float
    input_image_sha256: str
    validation_errors: tuple[str, ...]


class AnnotationValidationError(ValueError):
    """A model response does not satisfy the multiview annotation contract."""


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


def hash_composite_image(image: Image.Image) -> str:
    """Return a stable hash of the exact RGB composite supplied to the model."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def validate_annotation(annotation: Any) -> list[str]:
    """Return all structural and semantic validation errors for an annotation."""
    try:
        MultiviewAnnotation.model_validate(annotation)
    except ValidationError as error:
        messages: list[str] = []
        for detail in error.errors(include_url=False):
            location = ".".join(str(part) for part in detail["loc"])
            prefix = f"{location}: " if location else ""
            messages.append(f"{prefix}{detail['msg']}")
        return messages
    return []


def eligibility_reasons(annotation: dict[str, Any]) -> list[str]:
    """Explain why a valid annotation is unsuitable for LoRA caption training."""
    validation_errors = validate_annotation(annotation)
    if validation_errors:
        raise AnnotationValidationError("; ".join(validation_errors))

    reasons: list[str] = []
    for view in annotation["views"]:
        tile = view["tile"]
        for field in ("horizontal_view", "side", "vertical_angle"):
            if view[field] == "indeterminate":
                reasons.append(f"{tile}.{field} is indeterminate")
    return reasons


def _horizontal_phrase(horizontal: str, side: str) -> str:
    if horizontal == "side":
        return f"{side.title()} side"
    if horizontal == "front_three_quarter":
        return f"Front-{side} three-quarter"
    if horizontal == "rear_three_quarter":
        return f"Rear-{side} three-quarter"
    return {
        "front": "Front",
        "back": "Back",
        "top": "Top",
        "bottom": "Bottom",
    }[horizontal]


def render_caption(annotation: dict[str, Any]) -> str:
    """Render one eligible structured annotation into a deterministic caption."""
    reasons = eligibility_reasons(annotation)
    if reasons:
        raise AnnotationValidationError(
            "Cannot render an ineligible annotation: " + "; ".join(reasons)
        )

    vertical_phrases = {
        "overhead": "overhead",
        "high_angle": "high angle",
        "eye_level": "eye-level",
        "low_angle": "low angle",
        "underside": "underside",
    }
    parts = [f"[FOUR-VIEWS] Four views of {annotation['object_summary'].strip()}"]
    for view in annotation["views"]:
        tag = view["tile"].replace("_", "-").upper()
        viewpoint = _horizontal_phrase(view["horizontal_view"], view["side"])
        vertical = vertical_phrases[view["vertical_angle"]]
        features = ", ".join(feature.strip() for feature in view["visible_features"])
        parts.append(f"[{tag}] {viewpoint}, {vertical}; {features}")
    return "; ".join(parts) + "."


def generate_structured_annotation(
    composite_img_pil: Image.Image,
    category: str,
    *,
    model_id: str = MODEL_NAME,
    client: Any | None = None,
) -> CaptionResponse:
    """Request and validate a schema-constrained annotation for a 2x2 composite."""
    prompt = PROMPT_TEMPLATE.format(category=category)
    caption_client = client or create_client()
    image_hash = hash_composite_image(composite_img_pil)
    started = time.perf_counter()
    response = caption_client.models.generate_content(
        model=model_id,
        contents=[composite_img_pil, prompt],
        config=types.GenerateContentConfig(
            max_output_tokens=1000,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=MultiviewAnnotation,
        ),
    )
    latency_ms = (time.perf_counter() - started) * 1000
    raw_text = response.text or ""
    annotation: dict[str, Any] | None = None
    errors: list[str] = []
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as error:
        errors.append(f"response is not valid JSON: {error.msg}")
    else:
        try:
            annotation = MultiviewAnnotation.model_validate(parsed).model_dump(mode="json")
        except ValidationError:
            errors.extend(validate_annotation(parsed))

    return CaptionResponse(
        annotation=annotation,
        raw_response_text=raw_text,
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
        latency_ms=latency_ms,
        input_image_sha256=image_hash,
        validation_errors=tuple(errors),
    )
