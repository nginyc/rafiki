FROM node:11.1-alpine

ARG DOCKER_WORKDIR_PATH

RUN mkdir $DOCKER_WORKDIR_PATH
WORKDIR $DOCKER_WORKDIR_PATH

COPY web/package.json web/package.json
COPY web/package-lock.json web/package-lock.json

RUN cd web/ && npm install --production

COPY web/ web/

RUN cd web/ && npm run build

EXPOSE 3001

CMD ["node", "web/app.js"]