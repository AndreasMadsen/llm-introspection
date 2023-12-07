#!/bin/bash
source "jobs/_submitjob.sh"

declare -A time=( # ["llama2-70b IMDB"]="?:00" ["llama2-7b IMDB"]="?:00" ["falcon-40b IMDB"]="?:00" ["falcon-7b IMDB"]="?:00"
                    ["llama2-70b IMDB"]="2:00" ["llama2-7b IMDB"]="1:00" ["falcon-40b IMDB"]="1:00" ["falcon-7b IMDB"]="1:00"
                  # ["llama2-70b bAbI-1"]="?:00" ["llama2-7b bAbI-1"]="?:00" ["falcon-40b bAbI-1"]="?:00" ["falcon-7b bAbI-1"]="?:00"
                    ["llama2-70b bAbI-1"]="1:00" ["llama2-7b bAbI-1"]="1:00" ["falcon-40b bAbI-1"]="1:00" ["falcon-7b bAbI-1"]="1:00"
                  # ["llama2-70b bAbI-2"]="?:00" ["llama2-7b bAbI-1"]="?:00" ["falcon-40b bAbI-1"]="?:00" ["falcon-7b bAbI-1"]="?:00"
                    ["llama2-70b bAbI-2"]="1:00" ["llama2-7b bAbI-1"]="1:00" ["falcon-40b bAbI-1"]="1:00" ["falcon-7b bAbI-1"]="1:00"
                  # ["llama2-70b bAbI-3"]="?:00" ["llama2-7b bAbI-1"]="?:00" ["falcon-40b bAbI-1"]="?:00" ["falcon-7b bAbI-1"]="?:00"
                    ["llama2-70b bAbI-3"]="1:00" ["llama2-7b bAbI-1"]="1:00" ["falcon-40b bAbI-1"]="1:00" ["falcon-7b bAbI-1"]="1:00"
                  # ["llama2-70b MCTest"]="?:00" ["llama2-7b bAbI-1"]="?:00" ["falcon-40b bAbI-1"]="?:00" ["falcon-7b bAbI-1"]="?:00"
                    ["llama2-70b MCTest"]="0:30" ["llama2-7b bAbI-1"]="1:00" ["falcon-40b bAbI-1"]="1:00" ["falcon-7b bAbI-1"]="1:00" )

for model_name in 'llama2-70b' 'llama2-7b' 'falcon-40b' 'falcon-7b'
do
    for dataset in 'IMDB' 'bAbI-1' 'MCTest'
    do
        for system_message in 'none'
        do
            for task_config in '' 'c-persona-you' 'c-persona-human' 'm-removed' 'm-removed c-persona-you' 'm-removed c-persona-human' 'c-no-redacted' 'c-no-redacted c-persona-you' 'c-no-redacted c-persona-human'
            do
                submitjob "${time[$model_name $dataset]}" $(job_script tgi) \
                    experiments/analysis.py \
                    --task 'classify' \
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
