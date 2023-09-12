#!/bin/bash
#SBATCH -J unamed-instruct
#SBATCH --output=%x.%j.out
#SBATCH --account=rrg-bengioy-ad
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-gpu=32G
#SBATCH --time=2:59:00

# Check bash paramaters
if [ -z "${LOGDIR}" ]; then
    echo "LOGDIR enviorment variable was not provided" >&2
    exit 1
fi

# Init
module load gcc/9.3.0
nvidia-smi

# TGI config
declare -A model_id=( ["llama2-70b"]="meta-llama/Llama-2-70b-chat-hf"
                      ["llama2-13b"]="meta-llama/Llama-2-13b-chat-hf"
                      ["llama2-7b"]="meta-llama/Llama-2-7b-chat-hf"
                      ["falcon-40b"]="tiiuae/falcon-40b-instruct"
                      ["falcon-7b"]="tiiuae/falcon-7b-instruct" )
tgi_port=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
model_name=$(python -c 'import argparse; p = argparse.ArgumentParser(); p.add_argument("--model-name"); print(p.parse_known_args()[0].model_name)' "${@:2}")

# start TGI as a background process
MAX_CONCURRENT_REQUESTS=1024  MAX_INPUT_LENGTH=2048 \
    MAX_TOTAL_TOKENS=4096 MAX_BATCH_PREFILL_TOKENS=8192 \
    WAITING_SERVED_RATIO=1.2 MAX_WAITING_TOKENS=1024 \
    VALIDATION_WORKERS=4 PORT=$tgi_port \
    MODEL_ID="${model_id[$model_name]}" \
    bash tgi/tgi-server-cc.sh &> ${LOGDIR}/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.tgi &
TGI_PID=$!

# Create enviorment
module load gcc/9.3.0 python/3.11 git-lfs/3.3.0 cuda/11.8.0 cudnn/8.6.0.163 arrow/12.0.1
virtualenv --app-data $SCRATCH/virtualenv --no-download $SLURM_TMPDIR/pyenv
source $SLURM_TMPDIR/pyenv/bin/activate
python -m pip install --no-index -U pip
python -m pip install --no-index -U setuptools

# Install
cd $HOME/workspace/introspect
python -m pip install --no-index --find-links $HOME/python_wheels -e .

# Offline
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# Run
python -u -X faulthandler "$1" --persistent-dir $SCRATCH/introspect --endpoint "http://127.0.0.1:${tgi_port}" "${@:2}"

# Shutdown
kill -INT $TGI_PID