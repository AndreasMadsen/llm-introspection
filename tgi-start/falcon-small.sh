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

tgi_port=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "MODEL: tiiuae/falcon-7b-instruct"
echo "PROXY: ssh -N -f -L localhost:20002:$SLURMD_NODENAME:$tgi_port mila"
echo ""
echo ""

nvidia-smi

MAX_CONCURRENT_REQUESTS=1024 MAX_INPUT_LENGTH=2048 MAX_TOTAL_TOKENS=4096 MAX_BATCH_TOTAL_TOKENS=49152 \
    VALIDATION_WORKERS=4 PORT=$tgi_port MODEL_ID=tiiuae/falcon-7b-instruct \
    bash tgi/tgi-server-mila.sh
