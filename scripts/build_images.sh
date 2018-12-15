# Echo title with border
title() 
{
    title="| $1 |"
    edge=$(echo "$title" | sed 's/./-/g')
    echo "$edge"
    echo "$title"
    echo "$edge"
}

# Build Rafiki's images

title "Building Rafiki Admin's image..."
docker build -t $RAFIKI_IMAGE_ADMIN:$RAFIKI_VERSION -f ./dockerfiles/admin.Dockerfile \
    --build-arg DOCKER_WORKDIR_PATH=$DOCKER_WORKDIR_PATH \
    --build-arg CONDA_ENVIORNMENT=$CONDA_ENVIORNMENT $PWD || exit 1 
title "Building Rafiki Advisor's image..."
docker build -t $RAFIKI_IMAGE_ADVISOR:$RAFIKI_VERSION -f ./dockerfiles/advisor.Dockerfile \
    --build-arg DOCKER_WORKDIR_PATH=$DOCKER_WORKDIR_PATH \
    --build-arg CONDA_ENVIORNMENT=$CONDA_ENVIORNMENT $PWD || exit 1 
title "Building Rafiki Worker's image..."
docker build -t $RAFIKI_IMAGE_WORKER:$RAFIKI_VERSION -f ./dockerfiles/worker.Dockerfile \
    --build-arg DOCKER_WORKDIR_PATH=$DOCKER_WORKDIR_PATH \
    --build-arg CONDA_ENVIORNMENT=$CONDA_ENVIORNMENT $PWD || exit 1 
title "Building Rafiki Predictor's image..."
docker build -t $RAFIKI_IMAGE_PREDICTOR:$RAFIKI_VERSION -f ./dockerfiles/predictor.Dockerfile \
    --build-arg DOCKER_WORKDIR_PATH=$DOCKER_WORKDIR_PATH \
    --build-arg CONDA_ENVIORNMENT=$CONDA_ENVIORNMENT $PWD || exit 1 
title "Building Rafiki Admin Web's image..."
docker build -t $RAFIKI_IMAGE_ADMIN_WEB:$RAFIKI_VERSION -f ./dockerfiles/admin_web.Dockerfile \
    --build-arg DOCKER_WORKDIR_PATH=$DOCKER_WORKDIR_PATH $PWD || exit 1 
echo "Finished building all Rafiki's images successfully!"