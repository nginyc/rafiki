source .env

MODULE_NAME=atm_admin

docker rm -f $MODULE_NAME
docker build -t $MODULE_NAME -f admin.Dockerfile $PWD
docker run --name $MODULE_NAME \
  -v "/$PWD/admin/":"/root/app/admin/" \
  -v "/$PWD/atm/":"/root/app/atm/" \
  -v "/$PWD/prepare/":"/root/app/prepare/" \
  -v "/$PWD/data/":"/root/app/data/" \
  -v "/$PWD/scripts/":"/root/app/scripts/" \
  -v "/$PWD/models/":"/root/app/models/" \
  -v "/$PWD/methods/":"/root/app/methods/" \
  -e MYSQL_HOST=$MYSQL_HOST \
  -e MYSQL_PORT=$MYSQL_PORT \
  -e MYSQL_USER=$MYSQL_USER \
  -e MYSQL_DATABASE=$MYSQL_DATABASE \
  -e MYSQL_PASSWORD=$MYSQL_PASSWORD \
  -p 8000:8000 \
  $MODULE_NAME \
  bash -c "
    # Ensures python stdout appears immediately
    export PYTHONUNBUFFERED=1; 
    
    # Allow importing of python modules from project root
    export PYTHONPATH=/root/app/; 

    python scripts/start_admin.py;
  "