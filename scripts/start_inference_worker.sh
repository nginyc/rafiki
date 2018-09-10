MODULE_NAME=inference_worker

usage()  {
  echo "Usage: $0 <inference_job_id> <trial_id> <model_name>"
  exit 1
}

if [ $# -ne 3 ] ; then
    usage
fi

docker build -t $MODULE_NAME -f ./deploy/inference_worker.Dockerfile $PWD
docker run --rm --name $MODULE_NAME \
  --network $DOCKER_NETWORK \
  -e INFERENCE_JOB_ID=$1 \
  -e TRIAL_ID=$2 \
  -e MODEL_NAME=$3 \
  -e REDIS_HOST=$REDIS_HOST \
  -e REDIS_PORT=$REDIS_PORT \
  -e LOGS_FOLDER_PATH=$LOGS_FOLDER_PATH \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $LOGS_FOLDER_PATH:$LOGS_FOLDER_PATH \
  $MODULE_NAME