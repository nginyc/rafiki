IMAGE_NAME=rafiki_tf_model
DIR=$(dirname "$0")

docker build -t $IMAGE_NAME -f $DIR/../dockerfiles/TensorflowModel.Dockerfile $PWD