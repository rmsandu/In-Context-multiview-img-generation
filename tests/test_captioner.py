from types import SimpleNamespace

from PIL import Image

from src.captioner import MODEL_NAME, generate_caption_composite_grid


class FakeModels:
    def __init__(self):
        self.call = None

    def generate_content(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(text="[FOUR-VIEWS] caption")


def test_composite_caption_uses_expected_payload_and_config():
    models = FakeModels()
    client = SimpleNamespace(models=models)
    image = Image.new("RGB", (8, 8))

    caption = generate_caption_composite_grid(image, "bag", client=client)

    assert caption == "[FOUR-VIEWS] caption."
    assert models.call["model"] == MODEL_NAME
    assert models.call["contents"][0] is image
    assert "bag" in models.call["contents"][1]
    assert models.call["config"].temperature == 0.3
    assert models.call["config"].max_output_tokens == 200
