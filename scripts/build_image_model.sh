IMAGE_NAME=rafiki_model

docker build -t $IMAGE_NAME -f ./dockerfiles/model.Dockerfile $PWD