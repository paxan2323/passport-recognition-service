#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/passport-recognition-service}"
COMPOSE_FILES="-f docker-compose.yml -f docker-compose.prod.yml"
DOMAIN="${CERTBOT_DOMAIN:-pass.hermes-support.ru}"

if [ -f "/etc/letsencrypt/live/${DOMAIN}/fullchain.pem" ]; then
  COMPOSE_FILES="${COMPOSE_FILES} -f docker-compose.https.yml"
fi

cd "$APP_DIR"

echo "==> Pull latest code"
git fetch origin main
git reset --hard origin/main

echo "==> Build and start services"
docker compose $COMPOSE_FILES up -d --build --remove-orphans --force-recreate

echo "==> Wait for health"
for i in $(seq 1 30); do
  if curl -fsS http://127.0.0.1/health >/dev/null 2>&1; then
    echo "Health check OK"
    docker compose $COMPOSE_FILES ps
    exit 0
  fi
  sleep 5
done

echo "Health check failed"
docker compose $COMPOSE_FILES logs --tail=80 api worker nginx
exit 1
