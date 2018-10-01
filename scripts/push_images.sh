# Push Rafiki's images to Docker Hub

docker login

echo "Pushing all Rafiki's images to Docker Hub..."
docker push $RAFIKI_IMAGE_ADMIN:$RAFIKI_VERSION || exit 1 
docker push $RAFIKI_IMAGE_ADVISOR:$RAFIKI_VERSION || exit 1 
docker push $RAFIKI_IMAGE_WORKER:$RAFIKI_VERSION || exit 1 
docker push $RAFIKI_IMAGE_PREDICTOR:$RAFIKI_VERSION || exit 1 
echo "Pushed all images!"