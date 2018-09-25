source ./.env.sh

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
docker build -t $RAFIKI_IMAGE_ADMIN -f ./dockerfiles/admin.Dockerfile $PWD || exit 1 
title "Building Rafiki Advisor's image..."
docker build -t $RAFIKI_IMAGE_ADVISOR -f ./dockerfiles/advisor.Dockerfile $PWD || exit 1 
title "Building Rafiki Worker's image..."
docker build -t $RAFIKI_IMAGE_WORKER -f ./dockerfiles/worker.Dockerfile $PWD || exit 1 
title "Building Rafiki Query Frontend's image..."
docker build -t $RAFIKI_IMAGE_QUERY_FRONTEND -f ./dockerfiles/query_frontend.Dockerfile $PWD || exit 1 
echo "Finished building all Rafiki's images successfully!"