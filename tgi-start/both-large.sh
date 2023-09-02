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

echo "MODEL: tiiuae/falcon-40b-instruct"
echo "PROXY: ssh -N -f -L localhost:20001:$SLURMD_NODENAME:$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4)) mila"
echo ""
echo ""
echo "MODEL: meta-llama/Llama-2-70b-chat-hf"
echo "PROXY: ssh -N -f -L localhost:20002:$SLURMD_NODENAME:$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4)) mila"
echo ""
echo ""

nvidia-smi

MODEL_ID=tiiuae/falcon-40b-instruct TMP_PYENV=$SLURM_TMPDIR/tgl-env-01 SHARD_UDS_PATH=$SLURM_TMPDIR/tgl-server-socket-01 PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4)) MASTER_PORT=$(expr 30000 + $(echo -n $SLURM_JOBID | tail -c 4)) CUDA_VISIBLE_DEVICES='0,1' NUM_SHARD=2 bash tgi/tgi-server-mila.sh &
MODEL_ID=meta-llama/Llama-2-70b-chat-hf TMP_PYENV=$SLURM_TMPDIR/tgl-env-23 SHARD_UDS_PATH=$SLURM_TMPDIR/tgl-server-socket-23 PORT=$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4)) MASTER_PORT=$(expr 40000 + $(echo -n $SLURM_JOBID | tail -c 4)) CUDA_VISIBLE_DEVICES='2,3' NUM_SHARD=2 bash tgi/tgi-server-mila.sh &
wait
