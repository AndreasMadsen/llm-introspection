#!/bin/bash
#SBATCH -J tgi-server
#SBATCH --output=%x.%j.out
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=a100l:4
#SBATCH --ntasks=1
#SBATCH --constraint=ampere&nvlink
#SBATCH --mem=128G
#SBATCH --time=2:59:00
#SBATCH --partition=short-unkillable

echo "MODEL: meta-llama/Llama-2-70b-chat-hf"
echo "PROXY: ssh -N -f -L localhost:20002:$SLURMD_NODENAME:$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4)) mila"
echo ""
echo ""

nvidia-smi

export MAX_CONCURRENT_REQUESTS=256
export MAX_INPUT_LENGTH=2048
export MAX_TOTAL_TOKENS=4096
export MAX_BATCH_PREFILL_TOKENS=8192
export WAITING_SERVED_RATIO=1.2
export MAX_WAITING_TOKENS=20
export MAX_BATCH_TOTAL_TOKENS=65536
MODEL_ID=meta-llama/Llama-2-70b-chat-hf PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4)) bash tgi/tgi-server-mila.sh
