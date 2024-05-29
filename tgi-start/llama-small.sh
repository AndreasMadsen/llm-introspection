#!/bin/bash
#SBATCH -J tgi-server
#SBATCH --output=%x.%j.out
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=a100l.3:1
#SBATCH --ntasks=1
#SBATCH --constraint=ampere
#SBATCH --mem=24G
#SBATCH --time=2:59:00
#SBATCH --partition=unkillable

echo "MODEL: meta-llama/Llama-2-7b-chat-hf"
echo "PROXY: ssh -N -f -L localhost:20002:$SLURMD_NODENAME:$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4)) mila"
echo ""
echo ""

nvidia-smi

MODEL_PATH=/network/weights/llama.var/llama2/Llama-2-7b-chat-hf/ PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4)) bash tgi-sing/tgi-server-mila.sh
