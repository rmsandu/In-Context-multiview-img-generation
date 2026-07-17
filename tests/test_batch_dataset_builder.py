import json

from PIL import Image

from src.batch_dataset_builder import _batch_request, _caption_response, _response_text
from src.captioner import MAX_OUTPUT_TOKENS, PROMPT_VERSION


def test_batch_request_contains_image_prompt_and_structured_config():
    request = _batch_request(Image.new("RGB", (8, 8)), "bag", jpeg_quality=90)

    parts = request["contents"][0]["parts"]
    assert parts[0]["inlineData"]["mimeType"] == "image/jpeg"
    assert parts[0]["inlineData"]["data"]
    assert "bag" in parts[1]["text"]
    config = request["generationConfig"]
    assert config["maxOutputTokens"] == MAX_OUTPUT_TOKENS
    assert config["responseMimeType"] == "application/json"
    assert "additionalProperties" not in json.dumps(config["responseSchema"])
    assert "$defs" not in json.dumps(config["responseSchema"])
    assert "$ref" not in json.dumps(config["responseSchema"])
    assert config["thinkingConfig"]["thinkingBudget"] == 0


def test_batch_response_text_and_invalid_annotation_are_preserved():
    raw = _response_text({"candidates": [{"content": {"parts": [{"text": "not json"}]}}]})
    response = _caption_response(
        raw,
        {"composite_sha256": "a" * 64},
        "gemini-test",
    )

    assert response.raw_response_text == "not json"
    assert response.prompt_version == PROMPT_VERSION
    assert "not valid JSON" in response.validation_errors[0]
