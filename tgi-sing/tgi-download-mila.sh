#!/bin/bash
#SBATCH -J tgi-download
#SBATCH --output=%x.%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=1:00:00
set -e

# Default config
if [ -z "${RELEASE_DIR}" ]; then
    RELEASE_DIR=$HOME/tgi-sif
fi
if [ -z "${TGI_DIR}" ]; then
    TGI_DIR=$SCRATCH/tgi
fi
if [ -z "${TGI_TMP}" ]; then
    TGI_TMP=$SLURM_TMPDIR/tgi
fi

echo "Downloading ${MODEL_ID}"

# Load modules
module load apptainer

# Enable git-lfs support
eval "$(~/bin/micromamba shell hook -s posix)"
micromamba create -y -p $TGI_TMP/lfsenv -c conda-forge 'git-lfs=3.3'
micromamba activate $TGI_TMP/lfsenv

# prepear directories
mkdir -p $TGI_DIR/tgi-data
mkdir -p $TGI_DIR/tgi-repos

# download files
# huggingface_hub.download_snapshot is not used because the ignore_pattern does
# not support directories. So it downloads a ton of unused files.
if [[ -z "${HF_TOKEN}" ]]; then
  hf_url=https://huggingface.co
else
  hf_url=https://hf_user:${HF_TOKEN}@huggingface.co
fi

set +e  # ensure we reach `git remote rm origin`
if [ ! -d "${TGI_DIR}/tgi-repos/${MODEL_ID}" ] ; then
    GIT_LFS_SKIP_SMUDGE=1 git clone "${hf_url}/${MODEL_ID}" "${TGI_DIR}/tgi-repos/${MODEL_ID}"
    cd "${TGI_DIR}/tgi-repos/${MODEL_ID}"
    git remote rm origin
    git lfs install
fi

cd "${TGI_DIR}/tgi-repos/${MODEL_ID}"

# do not pull .bin files if .safetensors exists
git remote add origin "${hf_url}/${MODEL_ID}"
if ls *.safetensors 1> /dev/null 2>&1; then
  git lfs pull --exclude "*.bin,*.h5,*.msgpack,events.*,/logs,/coreml"
else
  git lfs pull --exclude "*.h5,*.msgpack,events.*,/logs,/coreml"
fi
git remote rm origin  # remove token reference

set -e

# convert .bin to .safetensors if needed
apptainer exec -C --nv \
    --env HF_HUB_DISABLE_TELEMETRY=1 --env HF_HUB_ENABLE_HF_TRANSFER=1 \
    --env HUGGINGFACE_HUB_CACHE=/tgi-cache \
    -W $SLURM_TMPDIR -B $TGI_DIR/tgi-data:/tgi-cache -B "${TGI_DIR}/tgi-repos/${MODEL_ID}:/tgi-model" \
    $RELEASE_DIR/tgi.sif text-generation-server \
    download-weights /tgi-model

echo "****************************"
echo "* DOWNLOAD JOB SUCCESSFULL *"
echo "****************************"
