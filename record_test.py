import sounddevice as sd
import numpy as np
import wave
import sys

TARGET_RATE = 16000
DURATION = 5 
FILENAME = "test_fixed.wav"
# Попробуй 15 (WASAPI) или None (Default)
INPUT_DEVICE = 15 

def main():
    print(f"--- Тестовая запись v2 ({DURATION} сек) ---")
    
    try:
        dev_info = sd.query_devices(INPUT_DEVICE, 'input')
        NATIVE_RATE = int(dev_info['default_samplerate'])
    except Exception as e:
        print(f"Ошибка устройства: {e}")
        return

    print(f"Микрофон: {dev_info['name']} ({NATIVE_RATE}Hz)")
    
    # Важно: записываем сразу ВЕСЬ блок на родной частоте
    print("\n🎤 ГОВОРИ СЕЙЧАС...")
    # Используем sd.rec, так как для разового теста это надежнее
    recording = sd.rec(int(DURATION * NATIVE_RATE), samplerate=NATIVE_RATE, 
                       channels=1, dtype='float32', device=INPUT_DEVICE)
    sd.wait() # Ждем завершения
    
    print("✅ Запись завершена. Обработка...")

    # Качественный ресемплинг всего массива сразу
    raw_audio = recording.flatten()
    num_samples_16k = int(len(raw_audio) * TARGET_RATE / NATIVE_RATE)
    indices = np.linspace(0, len(raw_audio) - 1, num_samples_16k)
    audio_16k = np.interp(indices, np.arange(len(raw_audio)), raw_audio)

    # Сохранение
    with wave.open(FILENAME, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(TARGET_RATE)
        # Нормализация уровня
        peak = np.max(np.abs(audio_16k))
        if peak > 0: audio_16k = audio_16k / peak * 0.9
        wf.writeframes((audio_16k * 32767).astype(np.int16).tobytes())
    
    print(f"Готово! Файл '{FILENAME}' можно проверять через curl.")

if __name__ == "__main__":
    main()