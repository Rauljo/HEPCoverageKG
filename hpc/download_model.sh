#!/bin/bash
#SBATCH -p COMPUTE
#SBATCH --exclude=compute-0-1
# compute-0-1 ACCEPTS WORK AND DOES NONE -- the same failure eval_job.sh has
# excluded since 2026-08-03, and the same shape as the faulty A100 (D-040).
# Without this line the first build sat RUNNING for 1h42 having not even
# written its output file, which is the tell: Slurm always creates one if the
# script actually started.
#SBATCH --job-name=hfget
#SBATCH --output=/home/xucabrjs/HEPCoverageKG/logs/hfget_%j.out
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
# =============================================================================
# Pull a model into the shared HF cache, ON A COMPUTE NODE.
#
# WHY A JOB AND NOT A SHELL COMMAND. These are 30-80GB and take tens of minutes
# to hours. The login node is capped and shared, and a download there is exactly
# the kind of long job that gets a user throttled.
#
# HF_HUB_OFFLINE IS FORCED OFF HERE and nowhere else. Every serving script sets
# it to 1 on purpose, so a job that would silently reach for the network fails
# fast instead. This is the one place the network is the point.
#
#   sbatch hpc/download_model.sh Qwen/Qwen3.8-27B
# =============================================================================
set -euo pipefail
MODEL="${1:?usage: download_model.sh <hf-repo-id>}"
module load Python/3.9.6-GCCcore-11.2.0
cd /home/xucabrjs/HEPCoverageKG
export HF_HUB_OFFLINE=0
# Only if the accelerator is actually installed. Setting it unconditionally
# does not fall back -- huggingface_hub RAISES ValueError and the job dies
# having transferred nothing.
if .venv/bin/python -c "import hf_transfer" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
    echo "hf_transfer: enabled"
else
    export HF_HUB_ENABLE_HF_TRANSFER=0
    echo "hf_transfer: not installed, using the standard downloader"
fi
echo "node: $(hostname)   model: ${MODEL}"
df -h "${HOME}" | tail -1
.venv/bin/python - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]
# allow_patterns: weights + config only. The repos carry .pth originals and
# duplicate formats that double the transfer for nothing.
path = snapshot_download(
    repo_id=repo,
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.py"],
    max_workers=8,
)
print("downloaded to:", path)
PY
du -sh ~/.cache/huggingface/hub/models--${MODEL//\//--} 2>/dev/null || true
echo "done"
