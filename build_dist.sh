#!/bin/bash

# Определяем разделитель для --add-data (в Windows ; в Linux :)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    SEP=";"
else
    SEP=":"
fi

echo "--- Начинаю сборку Jarvis Client ---"

# Проверяем наличие модели
if [ ! -f "silero_vad.onnx" ]; then
    echo "Ошибка: silero_vad.onnx не найден. Запустите entrypoint.sh или скачайте модель."
    exit 1
fi

# Сама команда сборки
# --onefile: всё в один файл
# --clean: очистить кэш перед сборкой
# --add-data: вшиваем модель внутрь
pyinstaller --onefile \
            --clean \
            --add-data "silero_vad.onnx${SEP}." \
            --name "JarvisClient" \
            client.py

echo "--- Сборка завершена! Ищи файл в папке /dist ---"