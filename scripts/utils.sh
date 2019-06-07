# Echo title with border
title() 
{
    title="| $1 |"
    edge=$(echo "$title" | sed 's/./-/g')
    echo "$edge"
    echo "$title"
    echo "$edge"
}

# Ensure docker container is stable
ensure_stable()
{
    LOG_FILE_PATH=$2
    SLEEP_TIME=$3
    echo "Waiting for ${SLEEP_TIME}s for $1 to stabilize..."
    sleep $SLEEP_TIME
    if ps -p $! > /dev/null
    then
        echo "$1 is running"
    else
        echo "Error running $1"
        echo "Maybe $1 hasn't previously been stopped - try running `scripts/stop.sh`?"
        if ! [ -z "$LOG_FILE_PATH" ]
        then
            echo "Check the logs at $LOG_FILE_PATH"
        fi
        exit 1
    fi
}