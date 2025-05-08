#!/bin/sh

HOST="$1"
PORT="$2"

echo "Menunggu database di $HOST:$PORT..."

while ! nc -z "$HOST" "$PORT"; do
  sleep 1
done

echo "Database siap!"
