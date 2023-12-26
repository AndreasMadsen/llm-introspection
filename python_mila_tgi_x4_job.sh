#!/bin/bash
#SBATCH -J unamed-introspect-job
#SBATCH --output=%x.%j.out
#SBATCH --constraint=ampere&nvlink
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-gpu=32G
#SBATCH --time=2:59:00
#SBATCH --partition=long

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
                      ["falcon-7b"]="tiiuae/falcon-7b-instruct"
                      ["mistral-v1-7b"]="mistralai/Mistral-7B-Instruct-v0.1"
                      ["mistral-v2-7b"]="mistralai/Mistral-7B-Instruct-v0.2" )
tgi_port=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
model_name=$(python -c 'import argparse; p = argparse.ArgumentParser(); p.add_argument("--model-name"); print(p.parse_known_args()[0].model_name)' "${@:2}")

# start TGI as a background process
MAX_CONCURRENT_REQUESTS=1024  MAX_INPUT_LENGTH=2048 MAX_TOTAL_TOKENS=4096 MAX_BATCH_TOTAL_TOKENS=49152 \
    VALIDATION_WORKERS=4 PORT=$tgi_port \
    MODEL_ID="${model_id[$model_name]}" \
    MAX_RESTARTS=5 bash monitor.sh bash tgi/tgi-server-mila.sh &> ${LOGDIR}/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.tgi &
TGI_PID=$!
echo "Started TGI server as background process [PID: ${TGI_PID}]"

# Create enviorment
eval "$(~/bin/micromamba shell hook -s posix)"
micromamba create -y -p $SLURM_TMPDIR/pyenv -c pytorch -c nvidia -c conda-forge 'python=3.11' 'git-lfs=3.3' 'pyarrow=12.0.1' 'pytorch==2.0.1' 'pytorch-cuda=11.8' 'cudnn=8.8' 'openssl=3'
micromamba activate $SLURM_TMPDIR/pyenv

# Install
cd $HOME/workspace/introspect
python -m pip install -e .

# Offline
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_DATASETS_CACHE="${SCRATCH}/introspect/cache/datasets"

# Run
python -u -X faulthandler "$1" --persistent-dir $SCRATCH/introspect --endpoint "http://127.0.0.1:${tgi_port}" "${@:2}"
PYTHON_EXIT_CODE=$?

# Shutdown
echo "Stopping TGI server as background process [PID: ${TGI_PID}]"
kill -TERM $TGI_PID

# finish
exit $PYTHON_EXIT_CODE
