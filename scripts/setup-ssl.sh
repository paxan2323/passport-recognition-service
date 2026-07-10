#!/usr/bin/env bash
set -euo pipefail

DOMAIN="${CERTBOT_DOMAIN:-pass.hermes-support.ru}"
EMAIL="${CERTBOT_EMAIL:-paxan232323@gmail.com}"
APP_DIR="${APP_DIR:-/opt/passport-recognition-service}"
WEBROOT="/var/www/certbot"

mkdir -p "$WEBROOT"

if [ ! -f "/etc/letsencrypt/live/${DOMAIN}/fullchain.pem" ]; then
  echo "==> Obtain initial certificate for ${DOMAIN}"
  apt-get update -qq
  apt-get install -y -qq certbot

  cd "$APP_DIR"
  # HTTP-only nginx for ACME webroot
  docker compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.ssl-bootstrap.yml up -d nginx

  certbot certonly --webroot \
    -w "$WEBROOT" \
    -d "$DOMAIN" \
    --email "$EMAIL" \
    --agree-tos \
    --non-interactive

  docker compose -f docker-compose.yml -f docker-compose.prod.yml down nginx
fi

echo "==> Certificate ready for ${DOMAIN}"
