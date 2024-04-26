#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH -J tgi-server
#SBATCH --output=%x.%j.out
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=2:59:00
set -e

if [ -z "${RELEASE_DIR}" ]; then
    RELEASE_DIR=$HOME/tgi-sif
fi
if [ -z "${TGI_DIR}" ]; then
    TGI_DIR=$SCRATCH/tgi
fi
if [ -z "${TGI_TMP}" ]; then
    TGI_TMP=$SLURM_TMPDIR/tgi
fi

mkdir -p $TGI_TMP

# Create env
module load StdEnv/2023
module load apptainer
rsync --archive --update --delete --verbose --human-readable --whole-file --inplace --no-compress --progress $RELEASE_DIR/ $TGI_TMP/env

# configure
default_port=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
default_master_port=$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4))
default_model_path=$TGI_DIR/tgi-repos/$MODEL_ID
default_num_shard=$(singularity exec -C --nv $TGI_TMP/env/tgi.sif python -c 'import torch; print(torch.cuda.device_count())')

# download model
rsync --archive --exclude='.git/' --update --delete --verbose --human-readable --whole-file --inplace --no-compress --progress ${MODEL_PATH:-$default_model_path}/ $TGI_TMP/model

# run model
apptainer run -C --nv \
    --env HF_HUB_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env HF_HUB_DISABLE_TELEMETRY=1 --env HF_HUB_ENABLE_HF_TRANSFER=1 \
    --env HUGGINGFACE_HUB_CACHE=/tgi-cache \
    -W $SLURM_TMPDIR -B $TGI_DIR/tgi-data:/tgi-cache -B $TGI_TMP/model:/tgi-model \
    $TGI_TMP/env/tgi.sif \
    --model-id /tgi-model \
    --num-shard "${NUM_SHARD:-$default_num_shard}" \
    --port "${PORT:-$default_port}" --master-port "${MASTER_PORT:-$default_master_port}"
