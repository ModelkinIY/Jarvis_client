import os
import io
import wave
import time
import queue
import threading
import json
import numpy as np
import requests
import sounddevice as sd
import onnxruntime as ort
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser 
from PIL import Image, ImageDraw
import pystray
from pystray import MenuItem as item
import pyperclip
import keyboard 
import soundfile as sf
import traceback
from datetime import datetime
import sys
import ai_module

def resource_path(relative_path):
    """ Получает абсолютный путь к ресурсам, работает для dev и для PyInstaller """
    try:
        # PyInstaller создает временную папку _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# И теперь обновите путь к модели:
VAD_MODEL_PATH = resource_path("silero_vad.onnx")


# --- КОНФИГУРАЦИЯ ---
CONFIG_FILE = "conf.json"
VAD_MODEL_PATH = "silero_vad.onnx"
TARGET_RATE = 16000

DEFAULT_CONFIG = {
    "stt_url": "http://127.0.0.1:8000/v1/audio/transcriptions",
    "stt_model": "large-v3-turbo",
    "llm_url": "http://127.0.0.1:11434/api/generate",
    "llm_model": "llama3",
    "tts_url": "http://127.0.0.1:8001/generate",
    "ai_activation_phrase": "джарвис",
    "ai_preprompt_path": "preprompt.txt",
    "ai_output_mode": "text_to_voice",
    "ai_hotkey_insert": "ctrl+shift+x",
    "hotkey_stt": "alt",
    "activation_mode": "hotkey", 
    "output_mode": "clipboard",
    "hotkey_insert": "ctrl+shift+v",
    "vad_threshold": 0.4,
    "silence_limit": 0.8,
    # Цвета состояний
    "color_idle_ptt": "#228B22",   # Зеленый
    "color_idle_voice": "#4169E1", # Синий
    "color_recording": "#FF4500",  # Оранжево-красный
    "color_stt": "#FFD700",        # Золотой
    "color_ai": "#9370DB",         # Пурпурный
    "color_tts": "#00CED1"         # Бирюзовый
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except: return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

def save_config(new_config):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_config, f, indent=4)

config = load_config()

# --- ГЛОБАЛЬНЫЕ ОБЪЕКТЫ ---
upload_q = queue.Queue()
raw_audio_q = queue.Queue() 
http_session = requests.Session()
running = True
root = None
gui_ready = threading.Event()
tray_icon = None

audio_buffer = []
is_recording = False
vad_thread_active = False
stt_shadow_buffer = ""
ai_shadow_buffer = ""

# --- УПРАВЛЕНИЕ СОСТОЯНИЯМИ И ЦВЕТОМ ---

def create_image(color):
    """Генерация квадратной иконки заданного цвета."""
    width, height = 64, 64
    image = Image.new('RGB', (width, height), color)
    dc = ImageDraw.Draw(image)
    dc.rectangle([8, 8, 56, 56], fill=None, outline="white", width=2)
    return image

def set_state_color(state_name):
    """Смена цвета иконки в трее."""
    global tray_icon
    if tray_icon:
        color_hex = config.get(f"color_{state_name}", "#FFFFFF")
        tray_icon.icon = create_image(color_hex)

def reset_to_idle():
    """Возврат в режим ожидания."""
    if config["activation_mode"] == "hotkey":
        set_state_color("idle_ptt")
    else:
        set_state_color("idle_voice")

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def play_tts_audio(wav_content):
    try:
        set_state_color("tts")
        if not wav_content or len(wav_content) < 100:
            print("[TTS] Ошибка: Некорректный аудио-файл.")
            reset_to_idle()
            return
        data, fs = sf.read(io.BytesIO(wav_content))
        print(f"[TTS] Воспроизведение ответа...")
        sd.play(data, fs)
        sd.wait()
        reset_to_idle()
    except Exception as e:
        print(f"[TTS] Ошибка воспроизведения: {e}")
        reset_to_idle()

class HotkeyEntry(ttk.Entry):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.bind("<KeyPress>", self._on_key_press)
        self.bind("<FocusIn>", lambda e: self.config(foreground="blue") if str(self['state']) != 'disabled' else None)
        self.bind("<FocusOut>", lambda e: self.config(foreground="black"))

    def _on_key_press(self, event):
        if str(self['state']) == 'disabled': return "break"
        combination = keyboard.get_hotkey_name()
        if combination:
            clean_hk = combination.replace(" ", "")
            self.delete(0, tk.END)
            self.insert(0, clean_hk)
        return "break"

# --- МОДУЛЬ ГОРЯЧИХ КЛАВИШ ---

def insert_stt_text():
    global stt_shadow_buffer
    if stt_shadow_buffer:
        print(f"[ВСТАВКА] STT текст...")
        keyboard.write(stt_shadow_buffer)
        stt_shadow_buffer = ""

def insert_ai_text():
    global ai_shadow_buffer
    if ai_shadow_buffer:
        print(f"[ВСТАВКА] AI ответ...")
        keyboard.write(ai_shadow_buffer)
        ai_shadow_buffer = ""

def bind_hotkeys():
    keyboard.unhook_all()
    if config["activation_mode"] == "hotkey":
        hk = config.get("hotkey_stt", "alt").lower().strip()
        try:
            if '+' in hk:
                keyboard.add_hotkey(hk, start_manual_record, args=(None,), trigger_on_release=False)
                keyboard.on_release_key(hk.split('+')[-1], stop_manual_record)
            else:
                keyboard.on_press_key(hk, start_manual_record)
                keyboard.on_release_key(hk, stop_manual_record)
            print(f"[СИСТЕМА] Хоткей записи: {hk}")
        except: pass

    if config["output_mode"] == "shadow_buffer":
        hk = config.get("hotkey_insert", "ctrl+shift+v").lower().strip()
        try: keyboard.add_hotkey(hk, insert_stt_text)
        except: pass

    if config["ai_output_mode"] == "shadow_buffer":
        hk = config.get("ai_hotkey_insert", "ctrl+shift+x").lower().strip()
        try: keyboard.add_hotkey(hk, insert_ai_text)
        except: pass

def start_manual_record(e):
    global is_recording, audio_buffer, stt_shadow_buffer, ai_shadow_buffer
    if not is_recording:
        set_state_color("recording")
        audio_buffer = []
        stt_shadow_buffer = ""
        ai_shadow_buffer = ""
        is_recording = True
        print("\n>>> [PTT] Слушаю...")

def stop_manual_record(e):
    global is_recording, audio_buffer
    if is_recording:
        is_recording = False
        if audio_buffer:
            print(f"<<< [PTT] Обработка...")
            upload_q.put(np.concatenate(audio_buffer))
        else:
            reset_to_idle()
        audio_buffer = []

# --- VAD WORKER ---
def vad_worker(native_rate):
    global vad_thread_active, stt_shadow_buffer, ai_shadow_buffer
    vad_thread_active = True
    try:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        session = ort.InferenceSession(VAD_MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
        h, c = np.zeros((2, 1, 64), dtype=np.float32), np.zeros((2, 1, 64), dtype=np.float32)
        chunk_size = int(512 * (native_rate / TARGET_RATE))
        phrase_acc, speaking, silence_cnt = [], False, 0
        buf = np.array([], dtype='float32')
        
        while running and vad_thread_active:
            if config["activation_mode"] != "voice": break
            try:
                data = raw_audio_q.get(timeout=0.5)
                buf = np.append(buf, data)
                while len(buf) >= chunk_size:
                    chunk = buf[:chunk_size]
                    buf = buf[chunk_size:]
                    if np.sqrt(np.mean(chunk**2)) < 0.002: prob = 0.0
                    else:
                        chunk_16k = np.interp(np.linspace(0, len(chunk)-1, 512), np.arange(len(chunk)), chunk)
                        out = session.run(None, {'input': chunk_16k.reshape(1, -1).astype(np.float32),
                                                 'sr': np.array(TARGET_RATE, dtype=np.int64), 'h': h, 'c': c})
                        prob, h, c = float(out[0].item()), out[1], out[2]
                    
                    if prob > config["vad_threshold"]:
                        if not speaking: 
                            set_state_color("recording")
                            print("\n>>> [VAD] Голос...")
                            stt_shadow_buffer = ""; ai_shadow_buffer = ""; speaking = True
                        phrase_acc.append(chunk); silence_cnt = 0
                    elif speaking:
                        phrase_acc.append(chunk); silence_cnt += 1
                        if silence_cnt > int(config["silence_limit"] * TARGET_RATE / 512):
                            print("<<< [VAD] Тишина. Отправка...")
                            upload_q.put(np.concatenate(phrase_acc))
                            phrase_acc, speaking = [], False
                            h, c = np.zeros((2, 1, 64), dtype=np.float32), np.zeros((2, 1, 64), dtype=np.float32)
            except queue.Empty: continue
    except Exception as e: print(f"[!] Ошибка VAD: {e}")
    finally: vad_thread_active = False

def apply_changes():
    global vad_thread_active
    print("[СИСТЕМА] Перезагрузка модулей...")
    while not raw_audio_q.empty(): raw_audio_q.get()
    bind_hotkeys()
    reset_to_idle()
    if config["activation_mode"] == "voice" and not vad_thread_active:
        dev = sd.query_devices(kind='input')
        rate = int(dev['default_samplerate'])
        threading.Thread(target=vad_worker, args=(rate,), daemon=True).start()

# --- GUI ---
def gui_thread_func():
    global root, gui_ready
    root = tk.Tk(); root.withdraw()
    gui_ready.set()
    root.mainloop()

def open_settings_window():
    if root: root.after(0, _create_settings_ui)

def _create_settings_ui():
    global config
    config = load_config()
    win = tk.Toplevel(root)
    win.title("Настройки Jarvis")
    win.attributes("-topmost", True)
    
    # Переменная для контроля единственного диалога выбора цвета
    _color_dialog_active = False

    main_container = ttk.Frame(win, padding="15")
    main_container.pack(fill="both", expand=True)
    main_container.columnconfigure(0, weight=1)

    # --- СЕКЦИИ STT, AI, TTS ---
    stt_group = ttk.LabelFrame(main_container, text=" Speech to Text ", padding="15")
    stt_group.grid(row=0, column=0, sticky="nsew", pady=(0, 10)); stt_group.columnconfigure(1, weight=1)
    
    ttk.Label(stt_group, text="Режим активации:").grid(row=0, column=0, sticky="w", pady=5)
    mode_c = ttk.Combobox(stt_group, values=["hotkey", "voice"], state="readonly")
    mode_c.set(config["activation_mode"]); mode_c.grid(row=0, column=1, sticky="ew", pady=5, padx=(10,0))

    ttk.Label(stt_group, text="Клавиша записи:").grid(row=1, column=0, sticky="w", pady=5)
    hk_r = HotkeyEntry(stt_group); hk_r.insert(0, config["hotkey_stt"]); hk_r.grid(row=1, column=1, sticky="ew", pady=5, padx=(10,0))

    ttk.Label(stt_group, text="Вывод результата:").grid(row=2, column=0, sticky="w", pady=5)
    out_m = {"shadow_buffer": "Shadow Buffer", "clipboard": "Буфер обмена", "direct_typing": "Прямая печать"}
    stt_o = ttk.Combobox(stt_group, values=list(out_m.values()), state="readonly")
    stt_o.set(out_m.get(config["output_mode"], "Буфер обмена")); stt_o.grid(row=2, column=1, sticky="ew", pady=5, padx=(10,0))

    ttk.Label(stt_group, text="Клавиша вставки STT:").grid(row=3, column=0, sticky="w", pady=5)
    hk_s = HotkeyEntry(stt_group); hk_s.insert(0, config["hotkey_insert"]); hk_s.grid(row=3, column=1, sticky="ew", pady=5, padx=(10,0))

    ttk.Label(stt_group, text="STT Server URL:").grid(row=4, column=0, sticky="w", pady=5)
    stt_u = ttk.Entry(stt_group); stt_u.insert(0, config["stt_url"]); stt_u.grid(row=4, column=1, sticky="ew", pady=5, padx=(10,0))

    ai_group = ttk.LabelFrame(main_container, text=" AI Assistant ", padding="15")
    ai_group.grid(row=1, column=0, sticky="nsew", pady=(0, 10)); ai_group.columnconfigure(1, weight=1)

    ttk.Label(ai_group, text="Вывод результата:").grid(row=0, column=0, sticky="w", pady=5)
    ai_m = {"text_to_voice": "Text to Voice (Голос)", "shadow_buffer": "Shadow Buffer (Вставка)"}
    ai_o = ttk.Combobox(ai_group, values=list(ai_m.values()), state="readonly")
    ai_o.set(ai_m.get(config["ai_output_mode"], "Text to Voice (Голос)")); ai_o.grid(row=0, column=1, sticky="ew", pady=5, padx=(10,0))

    ttk.Label(ai_group, text="Клавиша вставки AI:").grid(row=1, column=0, sticky="w", pady=5)
    hk_a = HotkeyEntry(ai_group); hk_a.insert(0, config["ai_hotkey_insert"]); hk_a.grid(row=1, column=1, sticky="ew", pady=5, padx=(10,0))

    ttk.Label(ai_group, text="Ollama URL:").grid(row=2, column=0, sticky="w", pady=5)
    llm_u = ttk.Entry(ai_group); llm_u.insert(0, config["llm_url"]); llm_u.grid(row=2, column=1, sticky="ew", pady=5, padx=(10,0))

    ttk.Label(ai_group, text="Model Name:").grid(row=3, column=0, sticky="w", pady=5)
    llm_m = ttk.Entry(ai_group); llm_m.insert(0, config["llm_model"]); llm_m.grid(row=3, column=1, sticky="ew", pady=5, padx=(10,0))

    ttk.Label(ai_group, text="Фраза активации:").grid(row=4, column=0, sticky="w", pady=5)
    ai_p = ttk.Entry(ai_group); ai_p.insert(0, config["ai_activation_phrase"]); ai_p.grid(row=4, column=1, sticky="ew", pady=5, padx=(10,0))

    tts_group = ttk.LabelFrame(main_container, text=" Text to Voice ", padding="15")
    tts_group.grid(row=2, column=0, sticky="nsew", pady=(0, 10)); tts_group.columnconfigure(1, weight=1)

    ttk.Label(tts_group, text="TTS Server URL:").grid(row=0, column=0, sticky="w", pady=5)
    tts_u = ttk.Entry(tts_group); tts_u.insert(0, config["tts_url"]); tts_u.grid(row=0, column=1, sticky="ew", pady=5, padx=(10,0))

    # --- STATE COLORS ---
    color_group = ttk.LabelFrame(main_container, text=" State Colors ", padding="15")
    color_group.grid(row=3, column=0, sticky="nsew", pady=(0, 10))
    color_group.columnconfigure(1, weight=1)

    colors = [
        ("Wait PTT", "color_idle_ptt"),
        ("Voice Listen", "color_idle_voice"),
        ("Recording", "color_recording"),
        ("Voice to Text", "color_stt"),
        ("AI Assistant", "color_ai"),
        ("Text to Voice", "color_tts")
    ]
    color_entries = {}

    def pick_color(key, entry_widget):
        nonlocal _color_dialog_active
        if _color_dialog_active:
            return
        
        _color_dialog_active = True
        current_color = entry_widget.get()
        
        # Временно отключаем topmost, чтобы системный диалог не ушел под окно
        win.attributes("-topmost", False)
        
        # Вызываем диалог. parent=win заставит его появиться поверх или рядом
        chosen = colorchooser.askcolor(initialcolor=current_color, title=f"Цвет: {key}", parent=win)
        
        # Возвращаем topmost
        win.attributes("-topmost", True)
        
        if chosen[1]:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, chosen[1].upper())
        
        _color_dialog_active = False

    for i, (label, cfg_key) in enumerate(colors):
        ttk.Label(color_group, text=f"{label}:").grid(row=i, column=0, sticky="w", pady=2)
        ent = ttk.Entry(color_group)
        ent.insert(0, config.get(cfg_key, "#FFFFFF"))
        ent.grid(row=i, column=1, sticky="ew", pady=2, padx=(10, 5))
        
        btn = ttk.Button(color_group, text="🎨", width=3, command=lambda k=label, e=ent: pick_color(k, e))
        btn.grid(row=i, column=2, sticky="e", pady=2)
        color_entries[cfg_key] = ent

    def update_ui(event=None):
        hk_r.config(state="disabled" if mode_c.get() == "voice" else "normal")
        hk_s.config(state="normal" if stt_o.get() == out_m["shadow_buffer"] else "disabled")
        hk_a.config(state="normal" if ai_o.get() == ai_m["shadow_buffer"] else "disabled")
    
    for w in [mode_c, stt_o, ai_o]: w.bind("<<ComboboxSelected>>", update_ui)
    update_ui()

    def save():
        config["activation_mode"], config["hotkey_stt"] = mode_c.get(), hk_r.get()
        config["hotkey_insert"], config["ai_hotkey_insert"] = hk_s.get(), hk_a.get()
        config["stt_url"], config["llm_url"] = stt_u.get(), llm_u.get()
        config["llm_model"], config["tts_url"] = llm_m.get(), tts_u.get()
        config["ai_activation_phrase"] = ai_p.get().lower()
        config["output_mode"] = {v: k for k, v in out_m.items()}.get(stt_o.get(), "clipboard")
        config["ai_output_mode"] = {v: k for k, v in ai_m.items()}.get(ai_o.get(), "text_to_voice")
        for key, entry in color_entries.items(): config[key] = entry.get()
        save_config(config); apply_changes(); win.destroy()

    ttk.Button(main_container, text="Сохранить настройки", command=save).grid(row=4, column=0, sticky="e", pady=10)

    # Авто-регулировка высоты и фиксация минимальных размеров
    win.update_idletasks()
    win.geometry("") # Сжимаем окно под контент
    # Фиксируем минимальную высоту по текущему контенту
    win.minsize(550, win.winfo_reqheight())

# --- WORKER ОБРАБОТКИ ---
def upload_worker(native_rate):
    global stt_shadow_buffer, ai_shadow_buffer
    while running:
        try:
            try:
                phrase_raw = upload_q.get(timeout=1)
            except queue.Empty: continue

            num_samples_16k = int((len(phrase_raw) / native_rate) * TARGET_RATE)
            audio_16k = np.interp(np.linspace(0, len(phrase_raw)-1, num_samples_16k), np.arange(len(phrase_raw)), phrase_raw)
            mx = np.max(np.abs(audio_16k))
            if mx < 0.01: 
                reset_to_idle()
                continue
            audio_int16 = (audio_16k / mx * 32767 * 0.9).astype(np.int16).tobytes()
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(TARGET_RATE)
                wf.writeframes(audio_int16)
            buf.seek(0)
            
            set_state_color("stt")
            start_stt = time.time()
            resp = http_session.post(config["stt_url"], files={'file': ('audio.wav', buf, 'audio/wav')}, 
                                     data={'model': config.get("stt_model", "large-v3-turbo"), 'language': 'ru'}, timeout=15)
            
            if resp.status_code == 200:
                text = resp.json().get('text', '').strip()
                if not text: 
                    reset_to_idle()
                    continue
                print(f"[STT] ({(time.time()-start_stt)*1000:.0f}ms) -> {text}")

                set_state_color("ai")
                ai_response = ai_module.process_ai_request(text, config)
                
                if ai_response:
                    if config["ai_output_mode"] == "text_to_voice":
                        print(f"[TTS] Синтез речи...")
                        t_resp = http_session.post(config["tts_url"], json={"text": ai_response, "speaker": "xenia"}, timeout=30)
                        if t_resp.status_code == 200: play_tts_audio(t_resp.content)
                        else: reset_to_idle()
                    else:
                        ai_shadow_buffer = ai_response
                        print(f"[AI] Сохранено в теневой буфер AI.")
                        reset_to_idle()
                else:
                    if config["output_mode"] == "clipboard": pyperclip.copy(text)
                    elif config["output_mode"] == "direct_typing": keyboard.write(text)
                    elif config["output_mode"] == "shadow_buffer": stt_shadow_buffer = text
                    reset_to_idle()
            else:
                reset_to_idle()
        except Exception:
            print(f"\n[КРИТИЧЕСКАЯ ОШИБКА ВОРКЕРА]\n{traceback.format_exc()}")
            reset_to_idle()
            time.sleep(1)

def on_quit(icon, item):
    global running, vad_thread_active
    print("[СИСТЕМА] Выход...")
    running = False; vad_thread_active = False
    keyboard.unhook_all()
    if root: root.after(0, root.quit)
    icon.stop()

def run_audio_engine():
    try:
        dev = sd.query_devices(kind='input')
        rate = int(dev['default_samplerate'])
        print(f"[СИСТЕМА] Микрофон: {dev['name']}")
        threading.Thread(target=upload_worker, args=(rate,), daemon=True).start()
        apply_changes() 
        with sd.InputStream(samplerate=rate, channels=1, callback=audio_callback, dtype='float32', blocksize=2048):
            while running: time.sleep(0.1)
    except Exception as e: print(f"[КРИТИЧЕСКАЯ ОШИБКА АУДИО] {e}")

def audio_callback(indata, frames, time, status):
    if running:
        if config["activation_mode"] == "hotkey":
            if is_recording: audio_buffer.append(indata.flatten().copy())
        else: raw_audio_q.put(indata.flatten().copy())

if __name__ == "__main__":
    print("--- Jarvis Client v1.3 (State Colors Fix) ---")
    threading.Thread(target=gui_thread_func, daemon=True).start()
    gui_ready.wait()
    
    initial_color = config.get("color_idle_ptt", "#228B22")
    tray_icon = pystray.Icon("Jarvis", create_image(initial_color), "Jarvis Voice")
    tray_icon.menu = pystray.Menu(item('Настройки', open_settings_window), item('Выход', on_quit))
    
    threading.Thread(target=run_audio_engine, daemon=True).start()
    tray_icon.run()