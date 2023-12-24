#!/bin/bash
declare -A idx=( ["IMDB"]="59" ["RTE"]="9" ["bAbI-1"]="55" )

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

for dataset in 'IMDB' 'RTE' 'bAbI-1'
do
    for system_message in 'none'
    do
        for task_config in '' 'c-persona-you e-persona-you' 'c-persona-human e-persona-human' 'e-implcit-target' 'e-implcit-target c-persona-you e-persona-you' 'e-implcit-target c-persona-human e-persona-human'
        do
            python export/latex_prompt.py \
                --task 'counterfactual' \
                --task-config $task_config \
                --system-message "${system_message}" \
                --dataset "${dataset}" \
                --split 'train' \
                --seed 0 \
                --idx "${idx[$dataset]}" &
        done

        for task_config in '' 'c-persona-you e-persona-you' 'c-persona-human e-persona-human' 'm-removed' 'm-removed c-persona-you e-persona-you' 'm-removed c-persona-human e-persona-human'
        do
            python export/latex_prompt.py \
                --task 'importance' \
                --task-config $task_config \
                --system-message "${system_message}" \
                --dataset "${dataset}" \
                --split 'train' \
                --seed 0 \
                --idx "${idx[$dataset]}" &
        done

        for task_config in '' 'c-persona-you e-persona-you' 'c-persona-human e-persona-human' 'm-removed' 'm-removed c-persona-you e-persona-you' 'm-removed c-persona-human e-persona-human'
        do
            python export/latex_prompt.py \
                --task 'redacted' \
                --task-config $task_config \
                --system-message "${system_message}" \
                --dataset "${dataset}" \
                --split 'train' \
                --seed 0 \
                --idx "${idx[$dataset]}" &
        done

        wait
    done
done
