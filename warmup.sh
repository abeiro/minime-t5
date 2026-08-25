#!/usr/bin/env bash

# Warm both inference paths so the first dialogue avoids model initialization.
if ! command -v curl >/dev/null 2>&1; then
    echo "$(date -Is) warm: curl unavailable; skipping model warmup"
    exit 0
fi

ready=0
ready_deadline=$((SECONDS + 90))
while ((SECONDS < ready_deadline)); do
    if curl --fail --silent --output /dev/null --connect-timeout 1 --max-time 2 \
        "http://127.0.0.1:8082/openapi.json"; then
        ready=1
        break
    fi
    sleep 2
done

if ((ready == 0)); then
    echo "$(date -Is) warm: Minime unavailable; skipping model warmup"
    exit 0
fi

if curl --fail --silent --show-error --output /dev/null --max-time 90 \
    "http://127.0.0.1:8082/detectMemory?text=warmup"; then
    echo "$(date -Is) warm: flan-t5 ready"
else
    echo "$(date -Is) warm: flan-t5 warmup failed"
fi

if curl --fail --silent --show-error --output /dev/null --max-time 60 \
    --request POST \
    --header "Content-Type: application/json" \
    --data '{"text":"warmup"}' \
    "http://127.0.0.1:8082/embed"; then
    echo "$(date -Is) warm: MiniLM ready"
else
    echo "$(date -Is) warm: MiniLM warmup failed"
fi
