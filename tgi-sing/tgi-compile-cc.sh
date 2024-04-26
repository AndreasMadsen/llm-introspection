#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH -J tgi-compile
#SBATCH --output=%x.%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=10:00:00
set -e
set -v

if [ -z "${RELEASE_DIR}" ]; then
    RELEASE_DIR=$HOME/tgi-sif
fi
if [ -z "${TGI_DIR}" ]; then
    TGI_DIR=$SCRATCH/tgi
fi
if [ -z "${TGI_TMP}" ]; then
    TGI_TMP=$SLURM_TMPDIR/tgi
fi

module load StdEnv/2023
module load apptainer
mkdir -p $RELEASE_DIR
apptainer build $RELEASE_DIR/tgi.sif docker://ghcr.io/huggingface/text-generation-inference:2.0.1
