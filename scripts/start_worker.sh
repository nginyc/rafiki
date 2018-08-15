source .env

MODULE_NAME=atm_worker

docker rm -f $MODULE_NAME
docker build -t $MODULE_NAME -f worker.Dockerfile $PWD
docker run --name $MODULE_NAME \
  -v "/$PWD/worker/":"/root/app/worker/" \
  -v "/$PWD/atm/":"/root/app/atm/" \
  -v "/$PWD/prepare/":"/root/app/prepare/" \
  -v "/$PWD/data/":"/root/app/data/" \
  -v "/$PWD/scripts/":"/root/app/scripts/" \
  -v "/$PWD/models/":"/root/app/models/" \
  -v "/$PWD/methods/":"/root/app/methods/" \
  -v "/$PWD/logs/":"/root/app/logs/" \
  -e MYSQL_HOST=$MYSQL_HOST \
  -e MYSQL_PORT=$MYSQL_PORT \
  -e MYSQL_USER=$MYSQL_USER \
  -e MYSQL_DATABASE=$MYSQL_DATABASE \
  -e MYSQL_PASSWORD=$MYSQL_PASSWORD \
  $MODULE_NAME \
  bash -c "
    # Ensures python stdout appears immediately
    export PYTHONUNBUFFERED=1; 
    
    # Allow importing of python modules from project root
    export PYTHONPATH=/root/app/; 

    python scripts/start_worker.py;
  "