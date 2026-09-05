#!/bin/bash
#SBATCH -p COMPUTE
#SBATCH --exclude=compute-0-1
# compute-0-1 ACCEPTS WORK AND DOES NONE -- the same failure eval_job.sh has
# excluded since 2026-08-03, and the same shape as the faulty A100 (D-040).
# Without this line the first build sat RUNNING for 1h42 having not even
# written its output file, which is the tell: Slurm always creates one if the
# script actually started.
#SBATCH --job-name=vllmimg
#SBATCH --output=/home/xucabrjs/HEPCoverageKG/logs/vllmimg_%j.out
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
# =============================================================================
# Build a newer vLLM apptainer image.
#
# WHY. The image we have is vLLM 0.8.5, and its model registry does not contain
# the architectures the recent Qwen models declare:
#
#   Qwen3.8-27B          Qwen3_5ForConditionalGeneration   NOT in 0.8.5
#   Qwen3.6-27B          Qwen3_5ForConditionalGeneration   NOT in 0.8.5
#   Qwen3-Coder-Next     Qwen3NextForCausalLM              NOT in 0.8.5
#   QwQ-32B-AWQ          Qwen2ForCausalLM                  supported -- which
#                                                          is why QwQ ran
#
# So ~180GB of downloaded weights are currently unservable, and the "is a newer
# model better at agentic work" question cannot be asked at all until this
# lands. That is the whole reason for the detour.
#
# ON A COMPUTE NODE, not the login node: this pulls ~10-16GB and unpacks it.
#
# THE DRIVER IS THE RISK, stated up front. The node runs 550.163.01 / CUDA 12.4.
# Recent vLLM ships CUDA 12.8/12.9 builds. CUDA minor-version compatibility
# normally lets those run on a 12.x driver, but if the image needs a driver
# newer than 550 it will fail at load, not at build -- so this job only BUILDS.
# Loading a model with it is a separate, cheap test.
# =============================================================================
set -euo pipefail
TAG="${1:-latest}"
cd /home/xucabrjs/hepcoveragekg_setup/images
# EVERY TEMP PATH ONTO /home, INCLUDING PLAIN TMPDIR.
#
# /home has 5.4TB free. Everything else -- /tmp, /, /export, /state/partition1
# -- is one 50GB volume with ~7GB free. The first attempt set APPTAINER_TMPDIR
# only, unpacked 32GB of layers to /home correctly, then sat in "Creating SIF
# file..." for 75 minutes without writing a byte: mksquashfs still had TMPDIR
# pointing at the 7GB volume and had nowhere to build a 20GB archive. It did not
# error, it just stopped, which is the worst shape a failure can take.
export APPTAINER_CACHEDIR=/home/xucabrjs/.apptainer_cache
export APPTAINER_TMPDIR=/home/xucabrjs/.apptainer_tmp
export TMPDIR="$APPTAINER_TMPDIR"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
echo "TMPDIR=$TMPDIR  ($(df -h "$TMPDIR" | tail -1 | awk '{print $4}') free)"
echo "node: $(hostname)   tag: ${TAG}"
df -h /home/xucabrjs | tail -1
# A SANDBOX, NOT A SIF. `--sandbox` writes a plain directory and skips
# mksquashfs altogether -- which is the step that stalled. It costs more disk
# (~32GB uncompressed against ~20GB) and /home has thousands of times that.
# `apptainer exec` runs a sandbox exactly as it runs a .sif, so nothing
# downstream changes except the path.
OUT="vllm-openai-${TAG}"
[ -e "$OUT" ] && { echo "$OUT already exists"; exit 0; }
apptainer build --sandbox --tmpdir "$APPTAINER_TMPDIR" \
    "$OUT" "docker://vllm/vllm-openai:${TAG}"
du -sh "$OUT"
cd /tmp && apptainer exec "/home/xucabrjs/hepcoveragekg_setup/images/$OUT" \
  python3 -c "
import torch, vllm
print('vllm', vllm.__version__)
# THE DRIVER IS THE BINDING CONSTRAINT, so it is reported first. The node runs
# 550.163.01 = CUDA 12.4. vLLM 0.28 is built against 12.8+ and torch refuses to
# initialise CUDA at all: 'The NVIDIA driver on your system is too old (found
# version 12040)'. The architecture check below is irrelevant unless this line
# reads 12.4 or lower.
print('torch', torch.__version__, 'built for CUDA', torch.version.cuda)
from vllm.model_executor.models.registry import ModelRegistry as R
a = R.get_supported_archs()
for arch in ('Qwen3_5ForConditionalGeneration','Qwen3NextForCausalLM','Qwen2ForCausalLM'):
    print(f'  {arch:38} {arch in a}')
"
echo done
