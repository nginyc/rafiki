usage()  {
  echo "Usage: $0 <container_name> <inference_job_id> <trial_id> <model_name>"
  exit 1
}

if [ $# -ne 4 ] ; then
    usage
fi

docker build -t $1 -f ./deploy/model.Dockerfile $PWD
docker run --rm --name $1 \
  --network $DOCKER_NETWORK \
  -e INFERENCE_JOB_ID=$2 \
  -e TRIAL_ID=$3 \
  -e MODEL_NAME=$4 \
  -e REDIS_HOST=$REDIS_HOST \
  -e REDIS_PORT=$REDIS_PORT \
  -e LOGS_FOLDER_PATH=$LOGS_FOLDER_PATH \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $LOGS_FOLDER_PATH:$LOGS_FOLDER_PATH \
  $1 \
  python start_inference_worker.py