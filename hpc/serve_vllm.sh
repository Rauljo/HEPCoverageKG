#!/bin/bash
# =============================================================================
# HEPCoverageKG: SLURM job serving Llama-3.1-8B-Instruct via self-hosted vLLM
# Run on the UCL Physics & Astronomy DIAS cluster (dias.hpc.phys.ucl.ac.uk).
#
# Model: NousResearch/Meta-Llama-3.1-8B-Instruct - an ungated mirror of
# meta-llama/Llama-3.1-8B-Instruct (identical weights), used to avoid waiting
# on Meta's manual gated-access approval.
#
# Image: vllm/vllm-openai:v0.8.5 - the newest vLLM release still built on
# CUDA 12.4.1. Newer tags (v0.9.0+) moved to CUDA 12.8.1, which this
# cluster's driver (550.163.01, CUDA 12.4 max) can't run - containers share
# the host's kernel driver, so the CUDA version in the image must not exceed
# what the driver supports.
#
# --mem 128G: a lower value (32G) caused "unable to mmap ... Cannot allocate
# memory" while loading the safetensors shards - mmap'd file pages count
# against the SLURM memory cgroup, so the limit needs enough headroom for
# all shards to be mapped at once, not just the model's final resident size.
#
# One-time setup (already done on this account, kept here for reference):
#   mkdir -p ~/hepcoveragekg_setup/{images,logs,scripts}
#   export APPTAINER_CACHEDIR=~/hepcoveragekg_setup/apptainer_cache
#   export APPTAINER_TMPDIR=~/hepcoveragekg_setup/apptainer_tmp
#   apptainer pull ~/hepcoveragekg_setup/images/vllm-openai-v0.8.5.sif \
#     docker://vllm/vllm-openai:v0.8.5
#
# Submit:
#   VLLM_API_KEY=<pick-a-key> sbatch hpc/serve_vllm.sh
#   # (or export VLLM_API_KEY in your shell first)
#
# Find the node it landed on, then tunnel in from anywhere you can SSH to
# the cluster's login node (compute nodes aren't directly reachable):
#   squeue -j <jobid> -h -o "%N"
#   ssh -f -N -L 8000:<node>:8000 dias
#   # then point HEPCoverageKG's .env at http://localhost:8000/v1
# =============================================================================
#SBATCH -p GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=vllm-hepckg
#SBATCH --output=/home/xucabrjs/hepcoveragekg_setup/logs/vllm_%j.out
#SBATCH --time=24:00:00
#SBATCH --mem 128G

: "${VLLM_API_KEY:?Set VLLM_API_KEY before submitting, e.g. VLLM_API_KEY=... sbatch hpc/serve_vllm.sh}"

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

apptainer exec --nv \
  ~/hepcoveragekg_setup/images/vllm-openai-v0.8.5.sif \
  vllm serve NousResearch/Meta-Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --api-key "$VLLM_API_KEY" \
  --gpu-memory-utilization 0.9
