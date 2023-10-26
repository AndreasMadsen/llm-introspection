#!/bin/bash

#
# This monitor program will restart the child command $MAX_RESTARTS times,
#  in case it crashes.
# If the monitor gets a SIGINT or SIGTERM it is forwarded as a SIGTERM to
#   child command and not restart the child command.
#
# Example:
#   MAX_RESTARTS=10 ./monitor.sh sleep 100
#

if [ -z "${MAX_RESTARTS}" ]; then
    MAX_RESTARTS=1
fi

should_restart=1
child_pid=0
function forward_signal()
{
    echo "Stop program from restarting"
    should_restart=0

    echo "Terminating program"
    if (( $child_pid > 0 )); then
        kill -TERM $child_pid
    fi

}

for (( restart_i=1; restart_i<=$MAX_RESTARTS; restart_i++ ));
do
    if (( $should_restart == 0 )); then
        break
    fi

    echo "Starting process, iteration $restart_i"

    "${@}" & # run arguments as command
    child_pid=$!
    trap forward_signal TERM INT
    wait $child_pid
    trap - TERM INT
    wait $child_pid
    child_pid=0
done
