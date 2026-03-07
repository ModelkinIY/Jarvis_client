import os
import sys
import io
import wave
import time
import queue
import threading
import numpy as np
import requests
import sounddevice as sd
import onnxruntime as ort

# --- КОНФИГУРАЦИЯ ---
# Используем прямой IP для исключения задержек DNS/IPv6 на Windows
SERVER_URL = "http://127.0.0.1:8000/v1/audio/transcriptions"
MODEL_NAME = "large-v3-turbo" 
VAD_MODEL_PATH = "silero_vad.onnx"

TARGET_RATE = 16000
VAD_THRESHOLD = 0.4
SILENCE_LIMIT = 0.8 
DEBUG_DIR = "debug_sent"

os.makedirs(DEBUG_DIR, exist_ok=True)

# Очереди для полной изоляции потоков
raw_audio_q = queue.Queue()
upload_q = queue.Queue()

# Единая сессия для Keep-Alive соединений
http_session = requests.Session()

# --- ИНИЦИАЛИЗАЦИЯ VAD ---
# Загружаем один раз при старте
vad_session = ort.InferenceSession(VAD_MODEL_PATH, providers=['CPUExecutionProvider'])
def reset_vad_state(): 
    return np.zeros((2, 1, 64), dtype=np.float32), np.zeros((2, 1, 64), dtype=np.float32)

def warm_up_server():
    """ 
    Выполняем только валидный POST-запрос с тишиной. 
    Это прогревает TCP-сессию и загружает веса модели в VRAM.
    """
    print("[СЕТЬ] Прогрев GPU и сетевого канала...")
    try:
        silent_data = io.BytesIO()
        with wave.open(silent_data, 'wb') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(TARGET_RATE)
            # Отправляем 0.2 сек тишины
            wf.writeframes(np.zeros(3200, dtype=np.int16).tobytes())
        silent_data.seek(0)
        
        files = {'file': ('warmup.wav', silent_data, 'audio/wav')}
        data = {'model': MODEL_NAME, 'language': 'ru'}
        # Первый запрос может занять время на аллокацию VRAM
        http_session.post(SERVER_URL, files=files, data=data, timeout=10)
        print("[СЕТЬ] Сервер готов к мгновенной обработке.")
    except Exception as e:
        print(f"[!] Ошибка прогрева (сервер еще запускается?): {e}")

# --- 1. ПОТОК ОТПРАВКИ (Изолирован от записи) ---
def upload_worker(native_rate):
    phrase_idx = 0
    while True:
        phrase_raw = upload_q.get()
        if phrase_raw is None: break
        
        phrase_idx += 1
        start_t = time.time()
        
        # Ресемплинг всей фразы целиком (исключает искажения на стыках)
        num_samples_16k = int((len(phrase_raw) / native_rate) * TARGET_RATE)
        indices = np.linspace(0, len(phrase_raw) - 1, num_samples_16k)
        audio_16k = np.interp(indices, np.arange(len(phrase_raw)), phrase_raw)
        
        # Нормализация громкости
        peak = np.max(np.abs(audio_16k))
        if peak > 0: audio_16k = audio_16k / peak * 0.9
        
        audio_int16 = (audio_16k * 32767).astype(np.int16).tobytes()

        # Дамп для аудита качества
        with wave.open(os.path.join(DEBUG_DIR, f"phrase_{phrase_idx:03d}.wav"), 'wb') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(TARGET_RATE)
            wf.writeframes(audio_int16)

        try:
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(TARGET_RATE)
                wf.writeframes(audio_int16)
            buf.seek(0)

            files = {'file': ('audio.wav', buf, 'audio/wav')}
            data = {'model': MODEL_NAME, 'language': 'ru'}
            
            resp = http_session.post(SERVER_URL, files=files, data=data, timeout=15)
            ms = (time.time() - start_t) * 1000
            
            if resp.status_code == 200:
                text = resp.json().get('text', '').strip()
                if text:
                    print(f"\n[{ms:.0f}ms] JARVIS: {text}")
            else:
                print(f"\n[!] Сервер вернул ошибку {resp.status_code}")
        except Exception as e:
            print(f"\n[!] Ошибка при передаче фразы: {e}")

# --- 1. ОПТИМИЗИРОВАННАЯ ИНИЦИАЛИЗАЦИЯ VAD ---
# --- ИНИЦИАЛИЗАЦИЯ VAD (Оптимизированная под 1 поток) ---
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

# Загружаем сессию с лимитом потоков
vad_session = ort.InferenceSession(VAD_MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])

def reset_vad_state(): 
    return np.zeros((2, 1, 64), dtype=np.float32), np.zeros((2, 1, 64), dtype=np.float32)

# --- 2. ПОТОК VAD С ЭНЕРГЕТИЧЕСКИМ ПОРОГОМ ---
def vad_worker(native_rate):
    h, c = reset_vad_state()
    vad_chunk_raw_size = int(512 * (native_rate / TARGET_RATE))
    
    phrase_acc = []
    is_speaking = False
    silence_count = 0
    vad_buf = np.array([], dtype='float32')
    
    # Порог громкости (0.002 - это очень тихий шум)
    # Если RMS ниже этого, мы даже не вызываем нейронку
    RMS_GATE = 0.002 

    while True:
        data = raw_audio_q.get()
        if data is None: break
        vad_buf = np.append(vad_buf, data)
        
        while len(vad_buf) >= vad_chunk_raw_size:
            chunk = vad_buf[:vad_chunk_raw_size]
            vad_buf = vad_buf[vad_chunk_raw_size:]
            
            # Считаем громкость чанка (RMS)
            rms = np.sqrt(np.mean(chunk**2))
            
            if rms < RMS_GATE:
                # Тишина - просто обнуляем вероятность, не нагружая CPU нейронкой
                prob = 0.0
            else:
                # Звук есть - запускаем VAD
                indices = np.linspace(0, len(chunk)-1, 512)
                chunk_16k = np.interp(indices, np.arange(len(chunk)), chunk)
                
                out = vad_session.run(None, {
                    'input': chunk_16k.reshape(1, -1).astype(np.float32),
                    'sr': np.array(TARGET_RATE, dtype=np.int64), 'h': h, 'c': c
                })
                prob, h, c = float(out[0].item()), out[1], out[2]

            # Дальше твоя стандартная логика без изменений
            if prob > VAD_THRESHOLD:
                if not is_speaking:
                    print("\n>>> СЛУШАЮ...")
                    is_speaking = True
                phrase_acc.append(chunk)
                silence_count = 0
            elif is_speaking:
                phrase_acc.append(chunk)
                silence_count += 1
                if silence_count > int(SILENCE_LIMIT * TARGET_RATE / 512):
                    print("\n<<< ОТПРАВКА НА СЕРВЕР...")
                    upload_q.put(np.concatenate(phrase_acc))
                    phrase_acc = []
                    is_speaking = False
                    h, c = reset_vad_state()

# --- 3. CALLBACK (Низкоуровневый захват) ---
def audio_callback(indata, frames, time, status):
    # Копируем данные в очередь немедленно, чтобы не блокировать поток аудиокарты
    raw_audio_q.put(indata.flatten().copy())

def main():
    print("--- Jarvis Client v6.9 (CPU Optimized Edition) ---")
    
    # 1. Сначала прогреваем сервер (подготавливаем TCP-сессию и VRAM)
    warm_up_server()

    # 2. Определяем параметры микрофона
    INPUT_DEVICE = 15  # Твой WASAPI ID
    try:
        dev_info = sd.query_devices(INPUT_DEVICE, 'input')
        NATIVE_RATE = int(dev_info['default_samplerate'])
    except Exception as e:
        print(f"[!] Ошибка доступа к устройству {INPUT_DEVICE}: {e}")
        # Если ID 15 не найден, пробуем устройство по умолчанию
        dev_info = sd.query_devices(kind='input')
        NATIVE_RATE = int(dev_info['default_samplerate'])
        INPUT_DEVICE = None

    print(f"Микрофон: {dev_info['name']}")
    print(f"Родная частота: {NATIVE_RATE}Hz | Целевая частота: {TARGET_RATE}Hz")

    # 3. Запускаем рабочие потоки (Upload и VAD)
    # Они будут висеть в фоне и ждать данных из очередей
    threading.Thread(target=vad_worker, args=(NATIVE_RATE,), daemon=True).start()
    threading.Thread(target=upload_worker, args=(NATIVE_RATE,), daemon=True).start()

    # 4. Основной цикл захвата звука
    try:
        # blocksize=2048 снижает частоту переключений контекста Python
        # Это критически важно для удержания загрузки CPU на уровне 5%
        with sd.InputStream(device=INPUT_DEVICE, 
                            samplerate=NATIVE_RATE, 
                            channels=1, 
                            callback=audio_callback, 
                            dtype='float32', 
                            blocksize=2048):
            
            print("\n>>> СИСТЕМА АКТИВНА. МОЖНО ГОВОРИТЬ.")
            print(">>> Нажмите Ctrl+C для выхода.")
            
            # В основном потоке просто спим. 
            # Вся работа происходит в callback и фоновых потоках.
            while True: 
                time.sleep(1.0) # Спим долго, не тратя такты процессора
                
    except KeyboardInterrupt:
        print("\n[.] Завершение работы по команде пользователя...")
    except Exception as e:
        print(f"\n[!] Критическая ошибка в потоке записи: {e}")
    finally:
        print("[.] Остановка аудиопотока. До связи!")

if __name__ == "__main__":
    main()