#!/bin/bash
source "jobs/_submitjob.sh"

declare -A time=( # ["llama2-70b IMDB"]="?:00" ["llama2-7b IMDB"]="?:00" ["falcon-40b IMDB"]="?:00" ["falcon-7b IMDB"]="?:00"
                    ["llama2-70b IMDB"]="14:00" ["llama2-7b IMDB"]="1:00" ["falcon-40b IMDB"]="1:00" ["falcon-7b IMDB"]="1:00"
                  # ["llama2-70b bAbI-1"]="?:00" ["llama2-7b bAbI-1"]="?:00" ["falcon-40b bAbI-1"]="?:00" ["falcon-7b bAbI-1"]="?:00"
                    ["llama2-70b bAbI-1"]="6:00" ["llama2-7b bAbI-1"]="1:00" ["falcon-40b bAbI-1"]="1:00" ["falcon-7b bAbI-1"]="1:00"
                  # ["llama2-70b MCTest"]="?:00" ["llama2-7b MCTest"]="?:00" ["falcon-40b MCTest"]="?:00" ["falcon-7b MCTest"]="?:00"
                    ["llama2-70b MCTest"]="6:00" ["llama2-7b MCTest"]="1:00" ["falcon-40b MCTest"]="1:00" ["falcon-7b MCTest"]="1:00")

for model_name in 'llama2-70b' 'llama2-7b' 'falcon-40b' 'falcon-7b'
do
    for dataset in 'IMDB' 'bAbI-1' 'MCTest'
    do
        for system_message in 'none'
        do
            for task_config in '' 'c-persona-you e-persona-you' 'c-persona-human e-persona-human' 'm-removed' 'm-removed c-persona-you e-persona-you' 'm-removed c-persona-human e-persona-human'
            do
                submitjob "${time[$model_name $dataset]}" $(job_script tgi) \
                    experiments/analysis.py \
                    --task 'redacted' \
                    --task-config $task_config \
                    --model-name "${model_name}" \
                    --system-message "${system_message}" \
                    --dataset "${dataset}" \
                    --split 'train' \
                    --seed 0 \
                    --clean-database
            done
        done
    done
    break
done
