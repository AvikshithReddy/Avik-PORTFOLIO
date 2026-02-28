FROM nginx:1.27-alpine

# Remove default server config
RUN rm /etc/nginx/conf.d/default.conf

# Custom Nginx config with backend proxy
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Static assets
COPY index.html /usr/share/nginx/html/index.html
COPY frontend/chatbot.js /usr/share/nginx/html/frontend/chatbot.js
COPY avatar.jpg /usr/share/nginx/html/avatar.jpg
COPY avikshithReddy_resume.pdf /usr/share/nginx/html/avikshithReddy_resume.pdf

EXPOSE 80
