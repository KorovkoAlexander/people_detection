FROM node:11-alpine

RUN apk add --update nginx

RUN npm config set unsafe-perm true

WORKDIR /tmp/nginx/detector_front
COPY package.json .
RUN npm install

COPY nginx/nginx.conf /etc/nginx/nginx.conf
COPY nginx/default.conf /etc/nginx/conf.d/default.conf

COPY . .
RUN npm run build
RUN mkdir -p /var/www/html
RUN cp -r dist/* /var/www/html

RUN mkdir -p /var/log/nginx
RUN chown nginx:nginx /var/www/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]