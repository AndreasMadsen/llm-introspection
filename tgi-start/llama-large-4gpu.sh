#!/bin/bash
#SBATCH -J tgi-server
#SBATCH --output=%x.%j.out
#SBATCH --constraint=ampere&nvlink
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-gpu=32G
#SBATCH --time=2:59:00
#SBATCH --partition=short-unkillable

tgi_port=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "MODEL: meta-llama/Llama-2-70b-chat-hf"
echo "PROXY: ssh -N -f -L localhost:20002:$SLURMD_NODENAME:$tgi_port mila"
echo ""
echo ""

nvidia-smi

MAX_CONCURRENT_REQUESTS=1024  MAX_INPUT_LENGTH=2048 \
    MAX_TOTAL_TOKENS=4096 MAX_BATCH_PREFILL_TOKENS=4096 \
    WAITING_SERVED_RATIO=1.2 MAX_WAITING_TOKENS=512 \
    VALIDATION_WORKERS=4 PORT=$tgi_port \
    MODEL_ID=meta-llama/Llama-2-70b-chat-hf bash tgi/tgi-server-mila.sh
