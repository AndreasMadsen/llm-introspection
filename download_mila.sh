#!/bin/bash
set -e

# Create enviorment
module load gcc/9.3.0
TMP_ENV=$(mktemp -d)
eval "$(~/bin/micromamba shell hook -s posix)"
micromamba create -y -p $TMP_ENV/pyenv -c pytorch -c conda-forge 'python=3.11' 'git-lfs=3.3' 'pyarrow=12.0.1' 'pytorch==2.0.1' 'openssl=3'
micromamba activate $TMP_ENV/pyenv

# Install
cd $HOME/workspace/introspect
python -m pip install -e .

# Fetch dataset
python experiments/download.py --persistent-dir $SCRATCH/introspect

# Check models
TGI_DIR=$SCRATCH/tgi
for model_id in 'meta-llama/Llama-2-70b-chat-hf' 'meta-llama/Llama-2-13b-chat-hf' 'meta-llama/Llama-2-7b-chat-hf' 'tiiuae/falcon-40b-instruct' 'tiiuae/falcon-7b-instruct' 'mistralai/Mistral-7B-Instruct-v0.2'
do
    if [ ! -d "${TGI_DIR}/tgi-repos/${model_id}" ] ; then
        echo "Model '${model_id}' is missing."
        echo "Consider running:"
        echo sbatch --export=ALL,HF_TOKEN=${HF_TOKEN},MODEL_ID=${model_id} tgi/tgi-download-mila.sh
    fi
done
