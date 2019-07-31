# /bin/bash

 docker run -e CHOKIDAR_USEPOLLING=true -e CHOKIDAR_INTERVAL=100 --name rafiki-web-ui -it -v ${PWD}:/app -p 3003:3001 rafiki-web-ui:dev 
