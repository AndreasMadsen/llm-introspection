#!/bin/bash
source "jobs/_submitjob.sh"

declare -A time=( # ["llama2-70b IMDB"]="?:00" ["llama2-7b IMDB"]="?:00" ["falcon-40b IMDB"]="?:00" ["falcon-7b IMDB"]="?:00"
                    ["llama2-70b IMDB"]="1:00" ["llama2-7b IMDB"]="1:00" ["falcon-40b IMDB"]="1:00" ["falcon-7b IMDB"]="1:00"
                  # ["llama2-70b SST2"]="?:00" ["llama2-7b SST2"]="?:00" ["falcon-40b SST2"]="?:00" ["falcon-7b SST2"]="?:00"
                    ["llama2-70b SST2"]="1:00" ["llama2-7b SST2"]="1:00" ["falcon-40b SST2"]="1:00" ["falcon-7b SST2"]="1:00" )

for model_name in 'llama2-70b' 'llama2-7b' 'falcon-40b' 'falcon-7b'
do
    for dataset in 'IMDB' 'SST2'
    do
        for system_message in 'none' 'default'
        do
            submitjob "${time[$model_name $dataset]}" $(job_script tgi) \
                experiments/answerable.py \
                --model-name "${model_name}" \
                --dataset "${dataset}" \
                --system-message "${system_message}" \
                --split 'train' \
                --seed 0 \
                --clean-database
            break
        done
        break
    done
    break
done