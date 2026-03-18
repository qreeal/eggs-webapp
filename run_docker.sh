#!/usr/bin/env bash

set -e

IMAGE_NAME="eggs-webapp"
CONTAINER_NAME="eggs-webapp-container"
PORT=5000

# Проверяем, установлен ли docker
if ! command -v docker >/dev/null 2>&1; then
  echo "❌ Docker не установлен. Установите Docker и запустите этот скрипт снова."
  exit 1
fi

echo "🔨 Собираем Docker-образ (${IMAGE_NAME})..."
docker build -t "${IMAGE_NAME}" .

# Если контейнер с таким именем уже крутится — остановим и удалим
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "🛑 Останавливаем старый контейнер ${CONTAINER_NAME}..."
  docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

echo "🚀 Запускаем контейнер..."
docker run -d --name "${CONTAINER_NAME}" -p ${PORT}:5000 "${IMAGE_NAME}"

URL="http://localhost:${PORT}"
echo "🌐 Приложение должно быть доступно по адресу: ${URL}"

# Попробуем автоматически открыть браузер
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "${URL}" >/dev/null 2>&1 || true
elif command -v open >/dev/null 2>&1; then
  open "${URL}" >/dev/null 2>&1 || true
fi
