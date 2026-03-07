#!/bin/bash
MODEL_PATH="/app/silero_vad.onnx"
MODEL_URL="https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx"

if [ ! -f "$MODEL_PATH" ]; then
    echo "--- Скачиваю модель VAD... ---"
    curl -L "$MODEL_URL" -o "$MODEL_PATH"
fi

echo "--- Подготовка окружения завершена ---"
# Мы НЕ запускаем тут python, чтобы контейнер не закрылся при ошибке скрипта