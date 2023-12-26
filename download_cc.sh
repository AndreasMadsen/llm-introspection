#!/bin/bash
set -e

# Create enviroment
module load gcc/9.3.0 python/3.11 git-lfs/3.3.0 arrow/12.0.1
TMP_ENV=$(mktemp -d)
virtualenv --app-data $SCRATCH/virtualenv --no-download $TMP_ENV/pyenv
source $TMP_ENV/pyenv/bin/activate
python -m pip install --no-index -U pip
python -m pip install --no-index -U setuptools

# Download package dependencies
mkdir -p $HOME/python_wheels
cd $HOME/python_wheels
pip download --no-deps 'datasets >= 2.14.6' 'tblib >= 2.0.0,<3.0.0' 'plotnine >= 0.12.0' 'mizani<0.10.0,>0.9.0' 'aiosqlite >= 0.19.0,<0.20.0' 'asyncstdlib >= 3.10.0,<4.0.0'

# Install project
cd $HOME/workspace/introspect
python -m pip install --no-index --find-links $HOME/python_wheels -e .

# Fetch dataset
python experiments/download.py --persistent-dir $SCRATCH/introspect

# Check models
TGI_DIR=$SCRATCH/tgi
for model_id in 'meta-llama/Llama-2-70b-chat-hf' 'meta-llama/Llama-2-13b-chat-hf' 'meta-llama/Llama-2-7b-chat-hf' 'tiiuae/falcon-40b-instruct' 'tiiuae/falcon-7b-instruct' 'mistralai/Mistral-7B-Instruct-v0.2' 'mistralai/Mistral-7B-Instruct-v0.1'
do
    if [ ! -d "${TGI_DIR}/tgi-repos/${model_id}" ] ; then
        echo "Model '${model_id}' is missing."
        echo "Consider running:"
        echo sbatch --export=ALL,HF_TOKEN=${HF_TOKEN},MODEL_ID=${model_id} tgi/tgi-download-cc.sh
    fi
done
