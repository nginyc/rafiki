FROM node:11.1-alpine

ARG DOCKER_WORKDIR_PATH

RUN mkdir $DOCKER_WORKDIR_PATH
WORKDIR $DOCKER_WORKDIR_PATH

COPY web/package.json web/package.json
COPY web/yarn.lock web/yarn.lock

RUN cd web/ && yarn install --production

COPY web/ web/

RUN cd web/ && yarn build

EXPOSE 3001

CMD ["node", "web/app.js"]