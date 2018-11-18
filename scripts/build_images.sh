# Echo title with border
title() 
{
    title="| $1 |"
    edge=$(echo "$title" | sed 's/./-/g')
    echo "$edge"
    echo "$title"
    echo "$edge"
}

build_image()
{
    docker build -t $1:$RAFIKI_VERSION -f $2 --build-arg DOCKER_WORKDIR_PATH=$DOCKER_WORKDIR_PATH $PWD || exit 1 
}

# Build Rafiki's images

title "Building Rafiki Admin's image..."
build_image $RAFIKI_IMAGE_ADMIN ./dockerfiles/admin.Dockerfile
title "Building Rafiki Advisor's image..."
build_image $RAFIKI_IMAGE_ADVISOR ./dockerfiles/advisor.Dockerfile
title "Building Rafiki Worker's image..."
build_image $RAFIKI_IMAGE_WORKER ./dockerfiles/worker.Dockerfile
title "Building Rafiki Predictor's image..."
build_image $RAFIKI_IMAGE_PREDICTOR ./dockerfiles/predictor.Dockerfile
title "Building Rafiki Admin Web's image..."
build_image $RAFIKI_IMAGE_ADMIN_WEB ./dockerfiles/admin_web.Dockerfile
echo "Finished building all Rafiki's images successfully!"