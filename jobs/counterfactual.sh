#!/bin/bash
source "jobs/_submitjob.sh"

declare -A gpus=( ["llama2-70b"]="x4" ["llama2-7b"]="x1" ["falcon-40b"]="x4" ["falcon-7b"]="x1" )
declare -A time=(   ["llama2-70b IMDB"]="10:00"  ["llama2-7b IMDB"]="1:00"   ["falcon-40b IMDB"]="1:00"   ["falcon-7b IMDB"]="1:00"
                    ["llama2-70b RTE"]="6:00"    ["llama2-7b RTE"]="1:00"    ["falcon-40b RTE"]="1:00"    ["falcon-7b RTE"]="1:00"
                    ["llama2-70b bAbI-1"]="6:00" ["llama2-7b bAbI-1"]="1:00" ["falcon-40b bAbI-1"]="1:00" ["falcon-7b bAbI-1"]="1:00"
                    ["llama2-70b MCTest"]="6:00" ["llama2-7b MCTest"]="1:00" ["falcon-40b MCTest"]="1:00" ["falcon-7b MCTest"]="1:00")

for model_name in 'llama2-70b' 'llama2-7b' 'falcon-40b' 'falcon-7b'
do
    for dataset in 'IMDB' 'RTE' 'bAbI-1' 'MCTest'
    do
        for system_message in 'none'
        do
            for task_config in '' 'c-persona-you e-persona-you' 'c-persona-human e-persona-human' 'e-implcit-target' 'e-implcit-target c-persona-you e-persona-you' 'e-implcit-target c-persona-human e-persona-human'
            do
                if [[ $model_name != 'llama2-70b' && $task_config != '' ]]; then
                    continue
                fi

                submitjob "${time[$model_name $dataset]}" $(job_script tgi ${gpus[$model_name]}) \
                    experiments/analysis.py \
                    --task 'counterfactual' \
                    --task-config $task_config \
                    --model-name "${model_name}" \
                    --system-message "${system_message}" \
                    --dataset "${dataset}" \
                    --split 'train' \
                    --seed 0
            done
        done
    done
done
