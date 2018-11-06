FROM node:11.1

RUN mkdir /root/rafiki/
WORKDIR /root/rafiki/

COPY web/ .

RUN npm install --production
RUN npm run build

EXPOSE 8080

ENTRYPOINT [ "node", "app.js" ]