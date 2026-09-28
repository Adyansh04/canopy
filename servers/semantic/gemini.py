"""Gemini's REST API as a describer, under client-side caps on each model's free tier."""

import copy
import fcntl
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .describers import Unavailable, http_post, jpeg_base64
from .prompts import answer_schema, parse_answer, prompt_for

PACIFIC = ZoneInfo("America/Los_Angeles")
# Google's free tier per model, [a minute, a day], from AI Studio's rate-limit page (2026-09-27).
# Config may lower a model's caps, never raise them; an unlisted model gets the smallest limits.
FREE_TIER = {
    "gemini-3.5-flash-lite": (15, 500),
    "gemini-3.1-flash-lite": (15, 500),
    "gemini-2.5-flash-lite": (10, 20),
    "gemma-4-26b-a4b-it": (30, 14400),
    "gemma-4-31b-it": (30, 14400),
}
FREE_TIER_UNLISTED = (5, 20)


class Capped(Unavailable):
    """Held back by the client's own caps, before any request was sent."""


def _next_pacific_day(now):
    day = datetime.fromtimestamp(now, PACIFIC).date() + timedelta(days=1)
    return datetime.combine(day, datetime.min.time(), PACIFIC).timestamp()


class GeminiLimiter:
    """Client-side caps on each Gemini model, shared by every process through a locked file.

    The day is Google's quota day, which turns over at midnight Pacific. Every request counts,
    failed ones too, and the count is taken before the request is sent.
    """

    def __init__(self, path, models, clock=time.time):
        self._path = Path(path).expanduser()
        self._caps = {}
        for model, caps in models.items():
            minute, day = FREE_TIER.get(model, FREE_TIER_UNLISTED)
            self._caps[model] = (
                min(int(caps["per_minute"]), minute),
                min(int(caps["per_day"]), day),
            )
        self._clock = clock

    def acquire(self, model):
        """Counts one request to `model`, or raises Unavailable without counting it."""
        per_minute, per_day = self._caps[model]

        def take(state, now):
            usage = state["models"].setdefault(
                model, {"count": 0, "recent": [], "parked_until": 0.0}
            )
            if now < usage["parked_until"]:
                raise Capped(f"parked for another {usage['parked_until'] - now:.0f} s")
            if usage["count"] >= per_day:
                raise Capped(f"its cap of {per_day} requests a day is reached")
            if len(usage["recent"]) >= per_minute:
                raise Capped(f"its cap of {per_minute} requests a minute is reached")
            usage["count"] += 1
            usage["recent"].append(now)

        self._update(take)

    def park(self, model, seconds=None):
        """No requests to `model` for `seconds`, or until the next Pacific day when None."""

        def change(state, now):
            usage = state["models"].setdefault(
                model, {"count": 0, "recent": [], "parked_until": 0.0}
            )
            until = _next_pacific_day(now) if seconds is None else now + seconds
            usage["parked_until"] = max(usage["parked_until"], until)

        self._update(change)

    def usage(self):
        return self._update(lambda state, now: copy.deepcopy(state))

    def _update(self, change):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # A separate lock file, because the state file is replaced rather than rewritten.
        with open(self._path.with_name(self._path.name + ".lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            now = self._clock()
            state = self._load(now)
            result = change(state, now)
            staging = self._path.with_name(self._path.name + ".tmp")
            staging.write_text(json.dumps(state))
            # Atomic, so a crash never leaves a truncated count that reads as zero.
            os.replace(staging, self._path)
            return result

    def _load(self, now):
        today = datetime.fromtimestamp(now, PACIFIC).date().isoformat()
        try:
            state = json.loads(self._path.read_text())
            models = {
                model: {
                    "count": int(usage["count"]) if state.get("day") == today else 0,
                    "recent": [float(stamp) for stamp in usage["recent"] if now - stamp < 60.0],
                    "parked_until": float(usage["parked_until"]),
                }
                for model, usage in state.get("models", {}).items()
            }
            if "models" not in state and state.get("day") == today:
                # Written before per-model caps: its one count and refusal hold for every model.
                old = {
                    "count": int(state.get("count", 0)),
                    "recent": [
                        float(stamp) for stamp in state.get("recent", []) if now - stamp < 60.0
                    ],
                    "parked_until": _next_pacific_day(now) if state.get("exhausted") else 0.0,
                }
                models = {model: copy.deepcopy(old) for model in self._caps}
        except FileNotFoundError:
            models = {}
        except (ValueError, TypeError, AttributeError, KeyError) as error:
            # Fail closed: an unreadable count must not read as a fresh day.
            raise Unavailable(
                f"{self._path} is unreadable ({error}); delete it to reset"
            ) from error
        return {"day": today, "models": models}


def _gemini_schema(schema):
    """JSON Schema to the OpenAPI subset that Gemini's responseSchema takes."""
    converted = {"type": schema["type"].upper()}
    if "enum" in schema:
        converted["enum"] = schema["enum"]
    if "properties" in schema:
        converted["properties"] = {
            key: _gemini_schema(value) for key, value in schema["properties"].items()
        }
        converted["required"] = schema["required"]
    return converted


def _read_key(path):
    """The key from a file holding it bare or as NAME=key; None when there is no file."""
    try:
        lines = Path(path).expanduser().read_text().splitlines()
    except FileNotFoundError:
        return None
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            return line.partition("=")[2].strip().strip("'\"") if "=" in line else line
    return None


def _retry_after(error):
    """Seconds a quota refusal asks to wait, or None for the rest of the Pacific day: any refusal
    that does not name only per-minute quotas, which is how the free tier was guarded before."""
    try:
        details = error.get("details") or []
        quotas = [v.get("quotaId", "") for d in details for v in d.get("violations", [])]
        delays = [float(d["retryDelay"].rstrip("s")) for d in details if "retryDelay" in d]
    except (AttributeError, TypeError, ValueError):
        return None
    if not quotas or not all("PerMinute" in quota for quota in quotas):
        return None
    return delays[0] if delays else 60.0


class GeminiDescriber:
    """Gemini's REST API with JSON output: the task's models in turn, each under its own caps."""

    name = "gemini"
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(
        self,
        routes,
        models,
        key_file,
        limiter,
        timeout_s=10.0,
        min_attempt_s=2.0,
        transport=http_post,
    ):
        for model in (model for route in routes.values() for model in route):
            if model not in models:
                raise ValueError(f"{model} is routed to but has no caps in gemini.models")
        self._routes = routes
        self._models = models
        self._key_file = key_file
        self._key = _read_key(key_file)
        self._limiter = limiter
        self._timeout_s = float(timeout_s)
        self._min_attempt_s = float(min_attempt_s)
        self._transport = transport

    def describe(self, task, images, context):
        if not self._key:
            raise Unavailable(f"no Gemini key in {self._key_file}")
        parts = [
            *(
                {"inlineData": {"mimeType": "image/jpeg", "data": jpeg_base64(image)}}
                for image in images
            ),
            {"text": prompt_for(task, len(images), context)},
        ]
        failures = []
        deadline = time.monotonic() + self._timeout_s
        for model in self._routes[task]:
            remaining = deadline - time.monotonic()
            if remaining < self._min_attempt_s:
                failures.append(f"{model}: out of time")
                break
            try:
                return self._ask(model, task, parts, remaining)
            except (Unavailable, ValueError) as error:
                failures.append(f"{model}: {error}")
                # A model at its cap is routine; one that was asked and failed is worth a line.
                if not isinstance(error, Capped):
                    print(f"  {model}: {error}", file=sys.stderr, flush=True)
        raise Unavailable("; ".join(failures) or f"no model for the {task} task")

    def _ask(self, model, task, parts, timeout):
        config = {
            "responseMimeType": "application/json",
            "responseSchema": _gemini_schema(answer_schema(task)),
            "mediaResolution": "MEDIA_RESOLUTION_LOW",
            "maxOutputTokens": 1024,
        }
        if self._models[model].get("thinking"):
            config["thinkingConfig"] = {"thinkingLevel": self._models[model]["thinking"]}
        body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": config}
        # Last step before the network, and the request counts even if the call then fails.
        self._limiter.acquire(model)
        # The key goes in a header only: URLs end up in logs and error messages.
        status, reply = self._transport(
            self.URL.format(model=model),
            {"x-goog-api-key": self._key, "Content-Type": "application/json"},
            json.dumps(body).encode(),
            timeout,
        )
        if status != 200:
            try:
                error = json.loads(reply).get("error", {})
            except (ValueError, AttributeError):
                error = {}
            if not isinstance(error, dict):  # {"error": "..."}, as a proxy in the way might send
                error = {"message": str(error)}
            detail = self._redact(f"{error.get('status', '')} {error.get('message', '')}".strip())
            if status == 429 or "RESOURCE_EXHAUSTED" in detail or "quota" in detail.lower():
                wait = _retry_after(error)
                self._limiter.park(model, wait)
                raise Unavailable(
                    f"refused on quota (HTTP {status}), parked "
                    + ("until midnight Pacific" if wait is None else f"for {wait:.0f} s")
                )
            # Busy (503) or broken (500): worth asking again next time.
            raise Unavailable(f"answered HTTP {status}: {detail[:200]}")
        data = json.loads(reply)
        candidate = (data.get("candidates") or [{}])[0]
        text = "".join(
            part.get("text", "")
            for part in candidate.get("content", {}).get("parts", [])
            if not part.get("thought")
        )
        answer = parse_answer(text, task)
        answer["model"] = data.get("modelVersion", model)
        return answer

    def _redact(self, text):
        return text.replace(self._key, "<key>") if self._key else text
