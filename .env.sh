# Core configuration for Rafiki
export DOCKER_NETWORK=rafiki
export RAFIKI_VERSION=0.0.9
export RAFIKI_IP_ADDRESS=127.0.0.1
export ADMIN_EXT_PORT=3000
export ADMIN_WEB_EXT_PORT=3001
export ADVISOR_EXT_PORT=3002
export POSTGRES_EXT_PORT=5433
export REDIS_EXT_PORT=6380

# Internal credentials for Rafiki's components
export POSTGRES_USER=rafiki
export POSTGRES_DB=rafiki
export POSTGRES_PASSWORD=rafiki

# Internal hosts & ports and configuration for Rafiki's components 
export POSTGRES_HOST=rafiki_db
export POSTGRES_PORT=5432
export ADMIN_HOST=rafiki_admin
export ADMIN_PORT=3000
export ADVISOR_HOST=rafiki_advisor
export ADVISOR_PORT=3002
export REDIS_HOST=rafiki_cache
export REDIS_PORT=6379
export PREDICTOR_PORT=3003
export ADMIN_WEB_HOST=rafiki_admin_web
export LOCAL_WORKDIR_PATH=$PWD
export DOCKER_WORKDIR_PATH=/root/rafiki
export CONDA_ENVIORNMENT=rafiki

# Docker images for Rafiki's custom components
export RAFIKI_IMAGE_ADMIN=rafikiai/rafiki_admin
export RAFIKI_IMAGE_ADMIN_WEB=rafikiai/rafiki_admin_web
export RAFIKI_IMAGE_ADVISOR=rafikiai/rafiki_advisor
export RAFIKI_IMAGE_WORKER=rafikiai/rafiki_worker
export RAFIKI_IMAGE_PREDICTOR=rafikiai/rafiki_predictor

# Docker images for dependent services
export IMAGE_POSTGRES=postgres:10.5-alpine
export IMAGE_REDIS=redis:5.0.3-alpine3.8

# Utility configuration
export PYTHONPATH=$PWD # Ensures that `rafiki` module can be imported at project root
