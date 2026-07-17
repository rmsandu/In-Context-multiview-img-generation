import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image

from src.captioner import (
    ANNOTATION_JSON_SCHEMA,
    HORIZONTAL_VIEWS,
    PROMPT_VERSION,
    AnnotationValidationError,
    MultiviewAnnotation,
    eligibility_reasons,
    generate_structured_annotation,
    render_caption,
    validate_annotation,
)


def valid_annotation():
    return {
        "object_summary": "blue canvas backpack with embroidered dragon",
        "views": [
            {
                "tile": "top_left",
                "horizontal_view": "rear_three_quarter",
                "side": "left",
                "vertical_angle": "eye_level",
                "framing": "full object",
                "visible_features": [
                    "left side pocket",
                    "two shoulder straps",
                    "partial dragon embroidery",
                ],
                "confidence": 0.83,
            },
            {
                "tile": "top_right",
                "horizontal_view": "side",
                "side": "left",
                "vertical_angle": "high_angle",
                "framing": "full object",
                "visible_features": ["side pocket", "zipper profile"],
                "confidence": 0.91,
            },
            {
                "tile": "bottom_left",
                "horizontal_view": "front",
                "side": "neither",
                "vertical_angle": "eye_level",
                "framing": "full object",
                "visible_features": ["dragon embroidery", "front pocket"],
                "confidence": 0.95,
            },
            {
                "tile": "bottom_right",
                "horizontal_view": "back",
                "side": "neither",
                "vertical_angle": "low_angle",
                "framing": "cropped object",
                "visible_features": ["two shoulder straps", "padded back"],
                "confidence": 0.88,
            },
        ],
    }


class FakeModels:
    def __init__(self, response_text):
        self.call = None
        self.response_text = response_text

    def generate_content(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(text=self.response_text)


def test_structured_caption_uses_expected_schema_and_payload():
    models = FakeModels(json.dumps(valid_annotation()))
    client = SimpleNamespace(models=models)
    image = Image.new("RGB", (8, 8))

    selected_model = "gemini-test-model"
    response = generate_structured_annotation(
        image, "bag", model_id=selected_model, client=client
    )

    assert response.annotation == valid_annotation()
    assert response.validation_errors == ()
    assert response.model_id == selected_model
    assert response.prompt_version == PROMPT_VERSION
    assert len(response.input_image_sha256) == 64
    assert response.latency_ms >= 0
    assert models.call["model"] == selected_model
    assert models.call["contents"][0] is image
    prompt = models.call["contents"][1]
    assert "bag" in prompt
    assert "absolute, static viewpoint" in prompt
    assert "can openers" in prompt
    config = models.call["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is MultiviewAnnotation
    assert config.temperature == 0.0
    assert config.max_output_tokens == 1000
    assert ANNOTATION_JSON_SCHEMA["properties"]["views"]["minItems"] == 4
    view_reference = ANNOTATION_JSON_SCHEMA["properties"]["views"]["items"]["$ref"]
    view_schema = ANNOTATION_JSON_SCHEMA["$defs"][view_reference.rsplit("/", 1)[-1]]
    horizontal_enum = view_schema["properties"]["horizontal_view"]["enum"]
    assert horizontal_enum == list(HORIZONTAL_VIEWS)
    assert "framing" in view_schema["required"]


def test_malformed_json_is_returned_with_validation_error_for_caching():
    models = FakeModels("not json")
    response = generate_structured_annotation(
        Image.new("RGB", (8, 8)), "tool", client=SimpleNamespace(models=models)
    )

    assert response.annotation is None
    assert response.raw_response_text == "not json"
    assert "not valid JSON" in response.validation_errors[0]


def test_render_caption_is_deterministic():
    caption = render_caption(valid_annotation())

    assert caption == (
        "[FOUR-VIEWS] Four views of blue canvas backpack with embroidered dragon; "
        "[TOP-LEFT] Rear-left three-quarter, eye-level; left side pocket, "
        "two shoulder straps, partial dragon embroidery; "
        "[TOP-RIGHT] Left side, high angle; side pocket, zipper profile; "
        "[BOTTOM-LEFT] Front, eye-level; dragon embroidery, front pocket; "
        "[BOTTOM-RIGHT] Back, low angle; two shoulder straps, padded back."
    )


def test_indeterminate_view_is_valid_but_not_eligible_or_renderable():
    annotation = valid_annotation()
    annotation["object_summary"] = "metal can opener with two handles"
    annotation["views"][0]["horizontal_view"] = "indeterminate"
    annotation["views"][0]["side"] = "indeterminate"

    assert validate_annotation(annotation) == []
    assert eligibility_reasons(annotation) == [
        "top_left.horizontal_view is indeterminate",
        "top_left.side is indeterminate",
    ]
    with pytest.raises(AnnotationValidationError, match="ineligible"):
        render_caption(annotation)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["views"].pop(), "at least 4 items"),
        (
            lambda value: value["views"][1].update(tile="top_left"),
            "tile values must be unique",
        ),
        (
            lambda value: value["views"][0].update(horizontal_view="diagonal"),
            "Input should be",
        ),
        (
            lambda value: value["views"][0].update(side="neither"),
            "must be left or right",
        ),
        (
            lambda value: value["views"][0].update(confidence=1.5),
            "less than or equal to 1",
        ),
        (lambda value: value["views"][0].pop("framing"), "Field required"),
    ],
)
def test_validation_rejects_invalid_annotations(mutation, message):
    annotation = deepcopy(valid_annotation())
    mutation(annotation)

    assert any(message in error for error in validate_annotation(annotation))
