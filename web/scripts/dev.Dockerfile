# base image
FROM node:12.2.0-alpine

# set working directory
WORKDIR /app

# add `/app/node_modules/.bin` to $PATH
ENV PATH /app/node_modules/.bin:$PATH

# install and cache app dependencies
COPY package.json /app/package.json
COPY yarn.lock /app/yarn.lock

RUN yarn install
RUN yarn global add react-scripts@3.0.1

EXPOSE 3001

# start app
CMD ["yarn", "start"]
