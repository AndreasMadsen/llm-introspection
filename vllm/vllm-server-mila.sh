#!/bin/bash
#SBATCH -J tgi-server
#SBATCH --output=%x.%j.out
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --constraint=ampere
#SBATCH --mem=24G
#SBATCH --time=2:59:00
set -e

VLLM_VERSION='0.1.4'

# Default config
if [ -z "${TGI_DIR}" ]; then
    TGI_DIR=$SCRATCH/tgi
fi
if [ -z "${TMP_PYENV}" ]; then
    TMP_PYENV=$SLURM_TMPDIR/tgl-env
fi

# Load modules
module load gcc/9.3.0

# Create enviorment
eval "$(~/bin/micromamba shell hook -s posix)"
micromamba create -y -p $TMP_PYENV -c pytorch -c nvidia -c conda-forge 'python=3.11' 'git-lfs=3.3' 'pyarrow=12.0.1' 'pytorch==2.0.1' 'pytorch-cuda=11.8' 'cudnn=8.8' 'openssl=3'
micromamba activate $TMP_PYENV

# install
pip install "ray[data]>=2.6.3,<3.0.0" "vllm==$VLLM_VERSION"

# configure
export HUGGINGFACE_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export HUGGINGFACE_HUB_CACHE=$TGI_DIR/tgi-data

export default_num_shard=$(python -c 'import torch; print(torch.cuda.device_count())')
export default_port=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export default_model_path=$TGI_DIR/tgi-repos/$MODEL_ID

echo "GPUS:" "${NUM_SHARD:-$default_num_shard}"

python -m vllm.entrypoints.api_server \
    --model "${MODEL_PATH:-$default_model_path}" \
    --tensor-parallel-size "${NUM_SHARD:-$default_num_shard}" \
    --port "${PORT:-$default_port}" \
    --host "0.0.0.0" \
    --download-dir $HUGGINGFACE_HUB_CACHE
