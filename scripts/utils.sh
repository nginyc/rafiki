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
    echo "Waiting for 10s for $1 to stablize..."
    sleep 10
    if ps -p $! > /dev/null
    then
        echo "$1 is running"
    else
        echo "Error running $1"
        if ! [ -z "$2" ]
        then
            echo "Check the logs at $2"
        fi
        exit 1
    fi
}