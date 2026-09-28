"""What a describer is asked, and its answer turned into the reply's fields."""

import functools
import json
import math
import re

from .config import REPO

ROOM_TABLE = REPO / "canopy/config/room_types.yaml"


@functools.cache
def _room_types():
    """The world model's room types: its table's, the hallway it types by shape, and 'other'."""
    import yaml

    return (*yaml.safe_load(ROOM_TABLE.read_text())["room_types"], "hallway", "other")


ROOM_SYNONYMS = {
    "lounge": "living room",
    "family room": "living room",
    "living": "living room",
    "dining": "dining room",
    "study": "office",
    "bath": "bathroom",
    "toilet": "bathroom",
    "restroom": "bathroom",
    "washroom": "bathroom",
    "storage": "storage room",
    "pantry": "storage room",
    "closet": "storage room",
    "garage": "storage room",
    "utility": "storage room",
    "corridor": "hallway",
    "hall": "hallway",
    "entry": "hallway",
    "foyer": "hallway",
}


def votes_in(context, key):
    """{name: count}, most votes first, from a dict, a list of names, or [name, count] pairs."""
    raw = context.get(key) or {}
    pairs = (
        raw.items()
        if isinstance(raw, dict)
        else (
            (entry[0], entry[1]) if isinstance(entry, (list, tuple)) else (entry, 1)
            for entry in raw
        )
    )
    votes = {}
    for name, count in pairs:
        name = " ".join(str(name).split())
        if name:
            votes[name] = votes.get(name, 0) + float(count)
    return dict(sorted(votes.items(), key=lambda item: -item[1]))


def _listing(votes):
    return ", ".join(f"{name} ({count:g})" for name, count in votes.items()) or "none"


def prompt_for(task, n_images, context):
    if task == "object":
        votes = votes_in(context, "labels")
        if not votes:
            raise ValueError("an object's context needs the detector's labels")
        views = "The image shows" if n_images == 1 else f"The {n_images} images show"
        # The detector's label comes last and only for label_ok: named up front, it anchors the
        # model, which then calls a fridge a cabinet because the detector did.
        # Never "a robot saw": shown too little, a small model answers with the prompt's words.
        return (
            f"{views} one indoor object, cut out of a camera image on a grey background.\n\n"
            "Answer in JSON:\n"
            "- name: what the object is, judged from the image alone: a common noun of 1 to 3 "
            'words, singular and lower case, such as "office chair", "mug" or "floor lamp".\n'
            "- caption: one short sentence on how it looks: colour, material, notable parts.\n"
            "- label_ok: an object detector, which is often wrong, called it "
            f'"{next(iter(votes))}". True if that is a fair name for what you see, else false.\n'
            "- confidence: 0 to 1, how sure you are of the name."
        )
    guess = room_type_of(context.get("room_type")) if context.get("room_type") else ""
    views = " The images are views from its camera." if n_images else ""
    return (
        f"A robot mapped one room of a home or office.{views} Objects found in it "
        f"(counts in brackets): {_listing(votes_in(context, 'objects'))}.\n\n"
        "Answer in JSON:\n"
        f"- room_type: one of {', '.join(_room_types())}.\n"
        '- name: a short name for the room, 1 to 3 words, such as "home office".\n'
        "- caption: one sentence on what the room is used for.\n"
        + (
            f'- label_ok: true if "{guess}" suits the room, else false.\n'
            if guess
            else "- label_ok: true.\n"
        )
        + "- confidence: 0 to 1, how sure you are of room_type."
    )


def answer_schema(task):
    properties = {
        "name": {"type": "string"},
        "caption": {"type": "string"},
        "label_ok": {"type": "boolean"},
        "confidence": {"type": "number"},
    }
    if task == "room":
        properties["room_type"] = {"type": "string", "enum": list(_room_types())}
    return {"type": "object", "properties": properties, "required": list(properties)}


def _noun(value):
    words = re.sub(r"[^a-z0-9\s-]", " ", str(value or "").lower()).split()
    if words and words[0] in ("a", "an", "the"):
        words = words[1:]
    # English puts the head noun last: "large red office chair" keeps "red office chair".
    return " ".join(words[-3:])


def room_type_of(value):
    text = " ".join(str(value or "").lower().replace("_", " ").replace("-", " ").split())
    for room in _room_types()[:-1]:  # not "other", which is inside words like "mother"
        if room in text:
            return room
    for word, room in ROOM_SYNONYMS.items():
        if word in text:
            return room
    return "other"


def _confidence(value):
    try:
        number = float(str(value).strip().rstrip("%"))
    except ValueError:
        return 0.5
    if not math.isfinite(number):
        return 0.5
    if number > 1.0:  # a percentage
        number /= 100.0
    return min(max(number, 0.0), 1.0)


def parse_answer(text, task):
    """A describer's JSON answer, normalised; ValueError when it holds nothing usable."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON in the answer: {text[:200]!r}")
    raw = json.loads(text[start : end + 1])
    if not isinstance(raw, dict):
        raise ValueError(f"the answer is not a JSON object: {text[:200]!r}")
    fields = {str(key).strip().lower(): value for key, value in raw.items()}
    name = _noun(fields.get("name"))
    room_type = room_type_of(fields.get("room_type")) if task == "room" else ""
    if task == "object" and not name:
        raise ValueError(f"the answer names no object: {text[:200]!r}")
    return {
        "name": name or room_type,
        "caption": " ".join(str(fields.get("caption") or "").split())[:300],
        "label_ok": str(fields.get("label_ok")).strip().lower() in ("true", "yes", "1"),
        "room_type": room_type,
        "confidence": _confidence(fields.get("confidence")),
    }
