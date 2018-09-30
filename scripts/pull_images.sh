pull_image()
{
    if [[ ! -z $(docker images -q $1) ]]
    then
        echo "$1 already exists locally"
    else 
        docker pull $1 || exit 1 
    fi
}

echo "Pulling images required by Rafiki from Docker Hub..."
pull_image $IMAGE_POSTGRES
pull_image $IMAGE_REDIS
pull_image $RAFIKI_IMAGE_ADMIN:$RAFIKI_VERSION
pull_image $RAFIKI_IMAGE_ADVISOR:$RAFIKI_VERSION
pull_image $RAFIKI_IMAGE_WORKER:$RAFIKI_VERSION
pull_image $RAFIKI_IMAGE_PREDICTOR:$RAFIKI_VERSION