read -p  "This machine's IP address in the network to expose Rafiki? " RAFIKI_IP_ADDRESS

cat > .env.sh << EOF
export POSTGRES_HOST=rafiki_db
export POSTGRES_PORT=5432
export POSTGRES_USER=rafiki
export POSTGRES_DB=rafiki
export POSTGRES_PASSWORD=rafiki
export APP_SECRET=rafiki
export DOCKER_NETWORK=rafiki
export LOGS_FOLDER_PATH=/var/log/rafiki
export ADMIN_HOST=rafiki_admin
export ADMIN_PORT=8000
export ADVISOR_HOST=rafiki_advisor
export ADVISOR_PORT=8001
export SUPERADMIN_EMAIL=superadmin@rafiki
export SUPERADMIN_PASSWORD=rafiki
export REDIS_HOST=rafiki_cache
export REDIS_PORT=6379
export QUERY_FRONTEND_PORT=8002
export REBROW_PORT=5001
export RAFIKI_IP_ADDRESS=$RAFIKI_IP_ADDRESS
export PYTHONPATH=$PWD
export RAFIKI_IMAGE_ADMIN=rafikiai/rafiki_admin
export RAFIKI_IMAGE_ADVISOR=rafikiai/rafiki_advisor
export RAFIKI_IMAGE_WORKER=rafikiai/rafiki_worker
export RAFIKI_IMAGE_QUERY_FRONTEND=rafikiai/rafiki_query_frontend
EOF

echo "Created $PWD/.env.sh"