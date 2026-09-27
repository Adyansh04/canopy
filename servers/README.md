# Model servers

The models canopy asks, served on the host GPU over ZMQ, because the robot's image carries no torch
or CUDA. Not a ROS package (`COLCON_IGNORE`): canopy_perception's `detector` and
`object_describer` are the clients.

| File | Does |
|---|---|
| `semantic_server.py` | Detection (YOLOE-26 over a word list, one forward pass for the whole list), image and text embeddings (SigLIP 2), and object and room descriptions (Gemini, falling back to a local VLM). REQ/REP on `tcp://127.0.0.1:5561`, msgpack bodies. |
| `start-vlm.sh` | The local VLM behind the `openai` describer: Qwen3.5-4B in llama.cpp's server, in a container, on `127.0.0.1:8080`. |
| `setup.sh` | A uv virtualenv with pinned torch and model libraries, the YOLOE weights, the Hugging Face downloads and the llama.cpp image. |
| `test_semantic_server.py` | Unit tests with fake models: no GPU, no network, no weights. |

## Setup and running

```bash
./servers/setup.sh                        # once; CANOPY_HOME (default ~/.local/share/canopy)
./servers/start-vlm.sh start              # the offline describer, about 4 GB of VRAM
~/.local/share/canopy/.venv/bin/python servers/semantic_server.py
```

YOLOE-26l and SigLIP 2 take about 1.7 GB of VRAM together. Each backend is a flag (`--detector`,
`--embedder`, `--describer`) or a `--config` YAML shaped like the server's `DEFAULTS`, and
`--self-test frame.png` runs one frame through without a socket.

## Endpoints

| Endpoint | In | Out |
|---|---|---|
| `ping` | | backend and device |
| `segment` | RGB image, phrases, thresholds, `embed` | instances: label, score, roi, mask crop, optional embedding |
| `embed_text`, `embed_image` | texts or images | unit vectors from the same model |
| `describe` | `task` (object or room), up to 8 images, what the map believes | name, caption, `label_ok`, `room_type`, confidence |

The server answers one request at a time, so a describe call of a few seconds delays the next
segment by as much.

## Gemini

The `gemini` describer needs a free API key in `~/.config/canopy/gemini.env` (the bare key, or
`NAME=key`). The key goes only in the `x-goog-api-key` header and is redacted from errors.

The free tier limits each model on its own, per Google Cloud project (AI Studio lists them under
Rate limits), so the server spreads its calls over several:

| Task | Models, in order | Free tier, each: a minute, a day |
|---|---|---|
| object | Gemini 3.5 Flash-Lite, Gemma 4 26B, Gemini 3.1 Flash-Lite | 15, 500; 30, 14,400; 15, 500 |
| room | Gemini 3.8 Flash, 3.6 Flash, then the first two above | 5, 20 for the Flash models |

- Each model has its own caps, a little under Google's, counted in a file beside the key across
  processes and restarts. `DEFAULTS["gemini"]` in `semantic_server.py` holds them.
- A model at its cap, refused or busy is skipped for the next. A refusal on the daily quota parks it
  until midnight Pacific; one on the minute quota, for the delay Google names.
- With every model out, the next describer answers: the local VLM by default.

One exploration of a six-room flat makes 100 to 250 describe calls, at most 13 in a minute.

How the models did on 12 crops of the flat's furniture, one per kind, with the object prompt:

| Model | Named right | Typical time |
|---|---|---|
| Gemini 3.5 Flash-Lite | 9 of 12 | 1.1-1.7 s |
| Gemini 3.1 Flash-Lite | 8 | 3 s, up to 14 s |
| Gemma 4 26B A4B | 8 | 2 s |
| Gemma 4 31B | 6 | 20-35 s, and HTTP 500s |
| Gemini 3.8 Live, extended thinking | 7 | 2 s |

The misses are mostly the same crops for every model: a daybed for the bed, a cabinet for the
wardrobe. The Live models have no daily limit, but they answer only in speech. The JSON read back
from the transcript was missing on 5 of those 12 (though on none of 20 retries), and each call
costs about 3,400 tokens of the 65,000 a minute. Gemma's 14,400 a day are plenty, so Live is not
used.

## Models and libraries

Each keeps its own license; follow the link.

| What | Used for | Source |
|---|---|---|
| YOLOE-26l-seg, -seg-pf (Ultralytics, AGPL-3.0) | detection | https://docs.ultralytics.com/models/yoloe (arXiv 2503.07465) |
| MobileCLIP2-B | YOLOE's text encoder | https://github.com/apple/ml-mobileclip |
| CLIP tokenizer (Ultralytics' fork) | YOLOE's text prompts | https://github.com/ultralytics/CLIP |
| SigLIP 2 B/16-256 | embeddings | https://huggingface.co/google/siglip2-base-patch16-256 |
| Gemini 3.5 and 3.1 Flash-Lite, 3.8 and 3.6 Flash | describer | https://ai.google.dev/gemini-api/docs/models |
| Gemma 4 26B A4B, through the Gemini API | describer | https://ai.google.dev/gemma |
| Qwen3.5-4B, Q4_K_M GGUF | local describer | https://huggingface.co/Qwen/Qwen3.5-4B, https://huggingface.co/unsloth/Qwen3.5-4B-GGUF |
| llama.cpp server | runs the local VLM | https://github.com/ggml-org/llama.cpp |
