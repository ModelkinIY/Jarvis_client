FROM python:3.11-slim

# Системные зависимости + инструменты разработки
RUN apt-get update && apt-get install -y \
    libportaudio2 \
    libasound2 \
    alsa-utils \
    curl \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Конфиг ALSA для WSLg
RUN echo 'pcm.!default { type pulse }' > /etc/asound.conf && \
    echo 'ctl.!default { type pulse }' >> /etc/asound.conf

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Скрипт просто будет лежать внутри, запускаем его через Compose
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh