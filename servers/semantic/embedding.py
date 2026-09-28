"""Embedders: `name`, embed_text(texts) and embed_images(images), each an L2-normalised
float32 (N, D) array."""

import numpy as np

# Mid grey is zero after SigLIP's normalisation, so masked-out pixels carry no signal.
GREY = 128


class Siglip2Embedder:
    """SigLIP 2: object crops and query text in one space, compared by dot product."""

    def __init__(self, device, model_id, revision):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.name = model_id
        self._device = device
        self._dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self._processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self._model = (
            AutoModel.from_pretrained(model_id, revision=revision, dtype=self._dtype)
            .to(device)
            .eval()
        )

    def embed_text(self, texts):
        # Trained on lower-case text padded to 64 tokens; other shapes shift the embedding.
        inputs = self._processor(
            text=[text.lower() for text in texts],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        return self._encode(
            self._model.get_text_features, input_ids=inputs["input_ids"].to(self._device)
        )

    def embed_images(self, images):
        # Arrays off the wire are read-only, and torch warns on wrapping them.
        images = [np.require(image, requirements="W") for image in images]
        inputs = self._processor(images=images, return_tensors="pt")
        return self._encode(
            self._model.get_image_features,
            pixel_values=inputs["pixel_values"].to(self._device, self._dtype),
        )

    @staticmethod
    def _encode(forward, **inputs):
        import torch

        with torch.inference_mode():
            output = forward(**inputs)
        features = getattr(output, "pooler_output", output)
        return torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy()


def masked_crop(image, roi, mask, pad=0.1):
    """The instance alone on a grey square with a margin: what gets embedded for it."""
    x, y, w, h = (int(value) for value in roi)
    side = int(round(max(w, h) * (1.0 + 2.0 * pad)))
    canvas = np.full((side, side, 3), GREY, np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    np.copyto(
        canvas[top : top + h, left : left + w],
        image[y : y + h, x : x + w],
        where=np.asarray(mask)[..., None] > 0,
    )
    return canvas
