#!/bin/bash
#SBATCH -p COMPUTE
#SBATCH --exclude=compute-0-1
#SBATCH --job-name=arm8b
#SBATCH --output=/home/xucabrjs/HEPCoverageKG/logs/arm8b_%j.out
#SBATCH --time=36:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
set -euo pipefail
module load Python/3.9.6-GCCcore-11.2.0
cd /home/xucabrjs/HEPCoverageKG
export LLM_BASE_URL="http://compute-gpu-0-1:8000/v1"
export CRITIC_BASE_URL="http://compute-gpu-0-1:8001/v1"
export CRITIC_MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"
echo "answerer=72B(8000)  judge=8B(8001)  questions=$1"
.venv/bin/python -m hepcoveragekg.cli eval run "$1" --system planner --repeats 3 --critic
