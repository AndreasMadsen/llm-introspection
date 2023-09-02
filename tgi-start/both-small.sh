#!/bin/bash
#SBATCH -J tgi-server
#SBATCH --output=%x.%j.out
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=a100l:1
#SBATCH --ntasks=1
#SBATCH --constraint=ampere
#SBATCH --mem=32G
#SBATCH --time=2:59:00
#SBATCH --partition=unkillable

echo "MODEL: tiiuae/falcon-7b-instruct"
echo "PROXY: ssh -N -f -L localhost:20001:$SLURMD_NODENAME:$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4)) mila"
echo ""
echo ""
echo "MODEL: meta-llama/Llama-2-7b-chat-hf"
echo "PROXY: ssh -N -f -L localhost:20002:$SLURMD_NODENAME:$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4)) mila"
echo ""
echo ""

nvidia-smi

MODEL_ID=tiiuae/falcon-7b-instruct TMP_PYENV=$SLURM_TMPDIR/tgl-env-falcon SHARD_UDS_PATH=$SLURM_TMPDIR/tgl-server-socket-falcon PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4)) MASTER_PORT=$(expr 30000 + $(echo -n $SLURM_JOBID | tail -c 4)) CUDA_MEMORY_FRACTION=0.5 bash tgi/tgi-server-mila.sh &
MODEL_ID=meta-llama/Llama-2-7b-chat-hf TMP_PYENV=$SLURM_TMPDIR/tgl-env-llama SHARD_UDS_PATH=$SLURM_TMPDIR/tgl-server-socket-llama PORT=$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4)) MASTER_PORT=$(expr 40000 + $(echo -n $SLURM_JOBID | tail -c 4)) CUDA_MEMORY_FRACTION=0.5 bash tgi/tgi-server-mila.sh &
wait
