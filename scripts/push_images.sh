source ./.env.sh

# Push Rafiki's images to Docker Hub

docker login

echo "Pushing all Rafiki's images to Docker Hub..."
docker push $RAFIKI_IMAGE_ADMIN || exit 1 
docker push $RAFIKI_IMAGE_ADVISOR || exit 1 
docker push $RAFIKI_IMAGE_MODEL || exit 1 
docker push $RAFIKI_IMAGE_QUERY_FRONTEND || exit 1 
echo "Pushed all images!"