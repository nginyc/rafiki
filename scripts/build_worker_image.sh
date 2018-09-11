IMAGE_NAME=rafiki_worker

docker build -t $IMAGE_NAME -f ./dockerfiles/worker.Dockerfile $PWD