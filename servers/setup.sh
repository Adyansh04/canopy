#!/usr/bin/env bash
#
# Sets up canopy's model servers on the host: servers/semantic_server.py (SAM 3.1 or YOLOE-26
# detector, SigLIP 2 embedder, Gemini or local VLM describer) and the local VLM that
# servers/start-vlm.sh runs in llama.cpp.
#
#   ./servers/setup.sh
#   CANOPY_HOME=/opt/canopy ./servers/setup.sh
#
# Idempotent. Safe to re-run to repair a half-finished install.
#
# Not a ROS package: the models need torch and CUDA, which the robot's image need not carry, so
# the server runs on the host and canopy_perception's detector speaks its wire protocol. Point
# WHEEL_CACHE at a cache other installs share, and the torch wheels download once.
set -euo pipefail

CANOPY_HOME="${CANOPY_HOME:-${HOME}/.local/share/canopy}"
WHEEL_CACHE="${WHEEL_CACHE:-${HOME}/.cache/canopy/wheels}"
WEIGHTS="${CANOPY_HOME}/weights"
PYTHON_VERSION="3.12"

TORCH_INDEX="https://download.pytorch.org/whl/cu128"
TORCH_WHEEL="torch-2.9.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl"
VISION_WHEEL="torchvision-0.24.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl"

DEPS=(
    "ultralytics==8.4.162"
    "transformers==5.17.0"
    "accelerate==1.14.0"
    "pillow==11.3.0"
    "pyzmq==27.0.1"
    "msgpack==1.1.0"
    "numpy==2.2.6"
    "sentencepiece==0.2.2"
    "pyyaml==6.0.3"
    # YOLOE's text tokenizer. Ultralytics would pip-install it on first use, which the server
    # forbids (YOLO_AUTOINSTALL=False), so it is pinned here instead.
    "clip @ git+https://github.com/ultralytics/CLIP.git@a13192f8cb767260d7dfd98c843b0716593169e7"
    # What the sam3 package imports, which its own pins would not leave alone (numpy < 2).
    "timm==1.0.30"
    "iopath==0.1.10"
    "einops==0.8.2"
    "pycocotools==2.0.11"
    # sam3's model builder imports pkg_resources, which setuptools 81 removed.
    "setuptools==80.9.0"
)
SAM3="sam3 @ git+https://github.com/facebookresearch/sam3.git@2345a4ad109ac29c569da749c91d84f10dc08c40"

# SAM 3.1's checkpoint, gated: request access at https://huggingface.co/facebook/sam3.1, then
# `hf auth login`. curl falls back to IPv4 when IPv6 hangs, where hf waits out each timeout.
SAM31_REPO="facebook/sam3.1"
SAM31_FILE="sam3.1_multiplex.pt"
HF_TOKEN_FILE="${HF_TOKEN_PATH:-${HOME}/.cache/huggingface/token}"

# YOLOE-26 large, text-prompted and prompt-free, and the MobileCLIP2 text encoder that
# text prompting loads from the working directory.
YOLOE_RELEASE="https://github.com/ultralytics/assets/releases/download/v8.4.0"
YOLOE_FILES=(yoloe-26l-seg.pt yoloe-26l-seg-pf.pt mobileclip2_b.ts)

SIGLIP="google/siglip2-base-patch16-256"
# semantic/config.py's DEFAULTS["siglip2"]["revision"] loads the same one.
SIGLIP_REVISION="3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab"
GGUF_REPO="unsloth/Qwen3.5-4B-GGUF"
GGUF_REVISION="e87f176479d0855a907a41277aca2f8ee7a09523"
GGUF_FILES=(Qwen3.5-4B-Q4_K_M.gguf mmproj-F16.gguf)
# One pin for the image, kept in start-vlm.sh.
LLAMA_IMAGE="$(grep -o 'ghcr.io/ggml-org/llama.cpp@sha256:[0-9a-f]*' "$(dirname "$0")/start-vlm.sh")"

for tool in uv git curl docker; do
    command -v "${tool}" >/dev/null || {
        echo "${tool} is not installed" >&2
        exit 1
    }
done

echo "==> torch wheels in ${WHEEL_CACHE}"
mkdir -p "${WHEEL_CACHE}"
for wheel in "${TORCH_WHEEL}" "${VISION_WHEEL}"; do
    name="${wheel//%2B/+}"
    if [ -s "${WHEEL_CACHE}/${name}" ]; then
        echo "    have ${name}"
        continue
    fi
    echo "    fetching ${name} (resumable; re-run this script if it stalls)"
    # Into .part first, so an interrupted download is resumed rather than taken as done.
    curl -fL -C - --retry 20 --retry-all-errors -o "${WHEEL_CACHE}/${name}.part" \
        "${TORCH_INDEX}/${wheel}"
    mv "${WHEEL_CACHE}/${name}.part" "${WHEEL_CACHE}/${name}"
done

echo "==> virtualenv at ${CANOPY_HOME}/.venv"
mkdir -p "${CANOPY_HOME}"
[ -d "${CANOPY_HOME}/.venv" ] || uv venv --python "${PYTHON_VERSION}" "${CANOPY_HOME}/.venv"

echo "==> installing torch, then the rest"
VIRTUAL_ENV="${CANOPY_HOME}/.venv" uv pip install --quiet \
    "${WHEEL_CACHE}/${TORCH_WHEEL//%2B/+}" "${WHEEL_CACHE}/${VISION_WHEEL//%2B/+}"
VIRTUAL_ENV="${CANOPY_HOME}/.venv" uv pip install --quiet \
    --index-url https://pypi.org/simple "${DEPS[@]}"
# --no-deps: both pin an old numpy, which nothing they use here needs.
VIRTUAL_ENV="${CANOPY_HOME}/.venv" uv pip install --quiet --no-deps msgpack-numpy==0.4.8 "${SAM3}"

echo "==> YOLOE-26 weights in ${WEIGHTS}"
mkdir -p "${WEIGHTS}"
for file in "${YOLOE_FILES[@]}"; do
    if [ -s "${WEIGHTS}/${file}" ]; then
        echo "    have ${file}"
        continue
    fi
    echo "    fetching ${file}"
    curl -fL -C - --retry 20 --retry-all-errors -o "${WEIGHTS}/${file}.part" \
        "${YOLOE_RELEASE}/${file}"
    mv "${WEIGHTS}/${file}.part" "${WEIGHTS}/${file}"
done

echo "==> SAM 3.1 checkpoint in ${WEIGHTS}"
if [ -s "${WEIGHTS}/${SAM31_FILE}" ]; then
    echo "    have ${SAM31_FILE}"
elif [ ! -s "${HF_TOKEN_FILE}" ]; then
    echo "    not fetched: no Hugging Face login (the weights are gated; see https://huggingface.co/${SAM31_REPO})"
else
    echo "    fetching ${SAM31_FILE} (3.5 GB)"
    # The token goes on stdin, never on a command line.
    printf 'Authorization: Bearer %s\n' "$(cat "${HF_TOKEN_FILE}")" |
        curl -fL -H @- -C - --retry 20 --retry-all-errors -o "${WEIGHTS}/${SAM31_FILE}.part" \
            "https://huggingface.co/${SAM31_REPO}/resolve/main/${SAM31_FILE}"
    mv "${WEIGHTS}/${SAM31_FILE}.part" "${WEIGHTS}/${SAM31_FILE}"
fi

echo "==> SigLIP 2 and Qwen3.5-4B GGUF in the Hugging Face cache"
"${CANOPY_HOME}/.venv/bin/hf" download "${SIGLIP}" --revision "${SIGLIP_REVISION}" >/dev/null
"${CANOPY_HOME}/.venv/bin/hf" download "${GGUF_REPO}" "${GGUF_FILES[@]}" \
    --revision "${GGUF_REVISION}" >/dev/null
echo "    have ${SIGLIP} and ${GGUF_FILES[*]}"

echo "==> llama.cpp server image"
if docker image inspect "${LLAMA_IMAGE}" >/dev/null 2>&1; then
    echo "    have ${LLAMA_IMAGE}"
else
    docker pull -q "${LLAMA_IMAGE}"
fi

echo "==> checking the install"
"${CANOPY_HOME}/.venv/bin/python" - <<'PY'
import clip  # noqa: F401
import msgpack_numpy  # noqa: F401
import torch
import zmq  # noqa: F401
from sam3.model_builder import build_sam3_image_model  # noqa: F401
from transformers import AutoModel  # noqa: F401
from ultralytics import YOLOE  # noqa: F401

print(f"    torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"    {torch.cuda.get_device_name(0)}, {total:.1f} GiB")
PY

cat <<EOF

Done. Serve the detector, embedder and describers on port 5561 (SAM 3.1 and SigLIP 2 use
about 8.5 GB of VRAM while mapping):

  ${CANOPY_HOME}/.venv/bin/python servers/semantic_server.py

YOLOE-26 in SAM 3.1's place takes about 1.7 GB with SigLIP 2, and its own word list
(canopy_perception's config/detector_yoloe.yaml):

  ${CANOPY_HOME}/.venv/bin/python servers/semantic_server.py --detector yoloe

Describers fall through in order, gemini,openai by default. Gemini reads its key from
~/.config/canopy/gemini.env and keeps each model under its free tier's limits a minute and a
day. The local VLM (Qwen3.5-4B in llama.cpp, about 4 GB of VRAM) serves the openai describer:

  ./servers/start-vlm.sh start      # stop when done: ./servers/start-vlm.sh stop
  ${CANOPY_HOME}/.venv/bin/python servers/semantic_server.py --describer openai

Check it against saved frames instead of serving:

  ${CANOPY_HOME}/.venv/bin/python servers/semantic_server.py --self-test frame.png

canopy_perception's detector, with its config/detector.yaml, asks it. Unit tests, no GPU needed:

  ${CANOPY_HOME}/.venv/bin/python -m unittest servers/test_semantic_server.py
EOF
