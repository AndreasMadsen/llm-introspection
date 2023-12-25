
PROJECT_RESULT_DIR="${SCRATCH}/introspect"
PROJECT_LOG_DIR="${PROJECT_RESULT_DIR}/logs"
if [ ! -z $SCRATCH ]; then
    mkdir -p $PROJECT_LOG_DIR
fi

function job_script {
    local cluster;

    if [ -z "${CC_CLUSTER}" ]; then
        cluster='mila';
    else
        cluster=$CC_CLUSTER;
    fi

    local jobscript="python_${cluster}_$1_$2_job.sh"

    if [ ! -f "$jobscript" ]; then
        echo "$jobscript not found" 1>&2
        return 1
    fi

    echo "$jobscript"
}

function join_by {
    local IFS="$1";
    shift;
    echo "$*";
}

function multiply_time {
    local walltime=$1;
    local multiplier=$2;

    python3 -c \
        "from datetime import timedelta; \
        in_h, in_m = '${walltime}'.split(':'); \
        t = timedelta(hours=int(in_h), minutes=int(in_m)) * ${multiplier}; \
        out_h, out_m = divmod(int(t.total_seconds()) // 60, 60); \
        print(f'{out_h:d}:{out_m:d}')";
}

function submitjob {
    local walltime=$1;
    local experiment_id;

    if [ ! -z $RUN_LOCALLY ]; then
        python "${@:3}"
        return 0;
    fi

    if ! experiment_id=$(python -m experiments.experiment_id "${@:3}"); then
        echo -e "\e[31mCould not get experiment name, error ^^^${experiment_id}\e[0m" >&2;
        return 1;
    fi

    if [[ $walltime == *"?"* ]]; then
        echo -e "\e[33mUndefined walltime $walltime for ${experiment_id}\e[0m" >&2;
        return 1;
    fi

    if [ ! -f "${PROJECT_RESULT_DIR}/results/${experiment_id%%_*}/${experiment_id}.json" ]; then
        echo "scheduling ${experiment_id}" 1>&2;

        if [ ! -z $RUN_DRY ]; then
            return 0;
        fi

        local jobid;
        if jobid=$(
            sbatch --time="$walltime:0" \
               --parsable \
               --export="ALL,LOGDIR=${PROJECT_LOG_DIR}" \
               -J "${experiment_id}" \
               -o "${PROJECT_LOG_DIR}/%x.%j.out" -e "${PROJECT_LOG_DIR}/%x.%j.err" \
               "${@:2}"
        ); then
            echo -e "\e[32msubmitted job as ${jobid}\e[0m" >&2
        else
            echo -e "\e[31mCould not submit ${experiment_id} with walltime '${walltime}', error ^^^${jobid}\e[0m" >&2
            return 1
        fi
    else
        echo -e "\e[34mskipping ${experiment_id}\e[0m" >&2
    fi
}
