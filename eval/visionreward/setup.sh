#!/usr/bin/env bash
# One-shot setup for the VisionReward-Video eval: create the conda env and fetch the
# 25 GB checkpoint. Idempotent — re-running skips whatever is already in place.
#
#   bash eval/visionreward/setup.sh              # env + weights
#   SKIP_WEIGHTS=1 bash eval/visionreward/setup.sh
#   SKIP_ENV=1     bash eval/visionreward/setup.sh
#
# Why a separate env: VisionReward-Video is CogVLM2-video via trust_remote_code, whose
# remote modeling code uses the legacy tuple KV-cache and hand-rolled 4-D masks that
# transformers removed in 4.47+. The `diffsynth` env runs transformers 5.9.0, so the
# model cannot load there. See requirements.txt.
set -euo pipefail

ENV_NAME="${ENV_NAME:-visionreward}"
CONDA_SH="${CONDA_SH:-/work/nlp/hzhao/miniforge3/etc/profile.d/conda.sh}"
MODEL_PATH="${MODEL_PATH:-/work/nlp/hzhao/checkpoints/visionreward/VisionReward-Video}"
MODEL_REPO="${MODEL_REPO:-THUDM/VisionReward-Video}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCRIPT_DIR  # read by the import-verification heredoc below

# Never let a user-site install leak in, in either direction: this box has a populated
# ~/.local/lib/python3.9/site-packages, and a bare `pip` there shadows the env's.
export PYTHONNOUSERSITE=1

PY=""

assert_env_python() {
    # Resolve the interpreter from CONDA_PREFIX and always drive pip as `$PY -m pip`.
    # A bare `pip` resolved to ~/.local/bin/pip on this machine and installed into (and
    # downgraded) the user's python3.9 site-packages instead of the env — hard to
    # notice, and it silently breaks unrelated tooling.
    PY="${CONDA_PREFIX:-}/bin/python"
    if [ -z "${CONDA_PREFIX:-}" ] || [ ! -x "$PY" ]; then
        echo "conda activate '$ENV_NAME' did not take effect (CONDA_PREFIX=" \
             "'${CONDA_PREFIX:-}'). Refusing to run pip against an unknown python." >&2
        exit 1
    fi
    case "$CONDA_PREFIX" in
        */envs/"$ENV_NAME") ;;
        *) echo "Active env is '$CONDA_PREFIX', expected .../envs/$ENV_NAME." >&2
           exit 1 ;;
    esac
    echo "==> using python: $PY"
}

# shellcheck disable=SC1090
source "$CONDA_SH"

if [ "${SKIP_ENV:-0}" != "1" ]; then
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "==> conda env '$ENV_NAME' already exists, reusing it"
    else
        echo "==> creating conda env '$ENV_NAME' (python $PYTHON_VERSION)"
        # `pip` is listed explicitly: without it the env can come up without a pip
        # module at all, and `python -m pip` then fails after activation.
        conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
    fi

    conda activate "$ENV_NAME"
    assert_env_python

    echo "==> installing torch $TORCH_VERSION / torchvision $TORCHVISION_VERSION"
    "$PY" -m pip install --index-url "$TORCH_INDEX" \
        "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"

    echo "==> installing eval requirements"
    "$PY" -m pip install -r "$SCRIPT_DIR/requirements.txt"
else
    conda activate "$ENV_NAME"
    assert_env_python
fi

if [ "${SKIP_WEIGHTS:-0}" != "1" ]; then
    # The safetensors index is the last file written, so its presence means a
    # previous download completed.
    if [ -f "$MODEL_PATH/model.safetensors.index.json" ] && \
       [ "$(find "$MODEL_PATH" -name 'model-*.safetensors' | wc -l)" -ge 6 ]; then
        echo "==> weights already at $MODEL_PATH"
    else
        echo "==> downloading $MODEL_REPO -> $MODEL_PATH (~25 GB)"
        mkdir -p "$MODEL_PATH"
        "$PY" - "$MODEL_REPO" "$MODEL_PATH" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo, local_dir = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo, local_dir=local_dir, max_workers=8,
                  ignore_patterns=["*.md", ".gitattributes"])
print("downloaded ->", local_dir)
PY
    fi
fi

echo
echo "==> verifying imports"
"$PY" - <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.environ["SCRIPT_DIR"]))
import torch, transformers
print("torch       ", torch.__version__)
print("transformers", transformers.__version__)
import decord; print("decord      ", decord.__version__)
from visionreward_video import install_video_transform_shims, load_checklist
install_video_transform_shims()
q, w = load_checklist()
print(f"checklist    {len(q)} questions / {len(w)} weights")
PY

echo
echo "Setup complete."
echo "  env:     $ENV_NAME"
echo "  weights: $MODEL_PATH"
echo
echo "Next: score a directory of clips, e.g."
echo "  conda activate $ENV_NAME"
echo "  bash eval/visionreward/score-demos.sh e2e-ttt-fomaml"
