"""Describers: `name` and describe(task, images, context) -> {"model", "name", "caption",
"label_ok", "room_type", "confidence"}, raising Unavailable (or anything) to fall through to the
next one in the chain."""

import base64
import http.client
import io
import json
import sys
import urllib.error
import urllib.request

import numpy as np

from .prompts import answer_schema, parse_answer, prompt_for, room_type_of, votes_in

# Longest side of an image sent to a describer: past this the tokens cost more than they tell.
MAX_IMAGE_SIDE = 512


class Unavailable(Exception):
    """A describer that cannot answer now: rate limited, out of quota, or unreachable."""


def http_post(url, headers, body, timeout):
    """(status, body) of a POST; Unavailable when the server cannot be reached at all."""
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
    except (OSError, http.client.HTTPException) as error:  # refused, reset, timed out, cut short
        raise Unavailable(f"{url.split('/')[2]} is unreachable: {error}") from error


def jpeg_base64(image):
    from PIL import Image

    picture = Image.fromarray(np.asarray(image, dtype=np.uint8))
    picture.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buffer = io.BytesIO()
    picture.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def padded_crop(image, roi, pad=0.1):
    """The instance's box and some surroundings: what a describer is shown."""
    x, y, w, h = (int(value) for value in roi)
    dx, dy = int(w * pad), int(h * pad)
    return image[max(y - dy, 0) : y + h + dy, max(x - dx, 0) : x + w + dx]


class OpenAIDescriber:
    """An OpenAI-compatible chat completions server, such as llama.cpp's on this host."""

    name = "openai"

    def __init__(self, base_url, model, timeout_s=60.0, transport=http_post):
        self.model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._timeout_s = float(timeout_s)
        self._transport = transport

    def describe(self, task, images, context):
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_base64(i)}"}}
            for i in images
        ]
        content.append({"type": "text", "text": prompt_for(task, len(images), context)})
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": 256,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": task, "schema": answer_schema(task)},
            },
        }
        status, reply = self._transport(
            self._url,
            {"Content-Type": "application/json"},
            json.dumps(body).encode(),
            self._timeout_s,
        )
        if status != 200:
            raise Unavailable(f"{self._url} answered HTTP {status}: {reply[:200]!r}")
        data = json.loads(reply)
        answer = parse_answer(data["choices"][0]["message"].get("content") or "", task)
        answer["model"] = data.get("model", self.model)
        return answer


class NoDescriber:
    """The detector's own label, or the caller's room guess, at confidence 0."""

    name = "none"

    def describe(self, task, images, context):
        del images
        if task == "object":
            votes = votes_in(context, "labels")
            if not votes:
                raise ValueError("an object's context needs the detector's labels")
            name, room_type = next(iter(votes)), ""
        else:
            # With no guess to echo, an empty type leaves the room to its objects.
            guess = context.get("room_type")
            name = room_type = room_type_of(guess) if guess else ""
        return {
            "model": "none",
            "name": name,
            "caption": "",
            "label_ok": True,
            "room_type": room_type,
            "confidence": 0.0,
        }


class DescriberChain:
    """Asks each describer in turn until one answers; the order is the preference."""

    def __init__(self, describers):
        self.describers = list(describers)

    @property
    def names(self):
        return [describer.name for describer in self.describers]

    def describe(self, task, images, context):
        failures = []
        for describer in self.describers:
            try:
                answer = describer.describe(task, images, context)
            except Exception as error:  # noqa: BLE001 - any failure moves on to the next one
                failures.append(f"{describer.name}: {error}")
                print(
                    f"  {describer.name} could not describe: {error}", file=sys.stderr, flush=True
                )
                continue
            answer["backend"] = describer.name
            return answer
        raise RuntimeError("no describer answered: " + ("; ".join(failures) or "none configured"))
