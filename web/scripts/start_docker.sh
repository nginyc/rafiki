# /bin/bash

 docker run -e CHOKIDAR_USEPOLLING=true -e CHOKIDAR_INTERVAL=100 --name singaauto-web-ui -it -v ${PWD}:/app -p 3003:3001 -p 9000:9000 singaauto-web-ui:dev /bin/sh
