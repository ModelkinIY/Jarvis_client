import os
import json
import requests
import traceback
from datetime import datetime

DEBUG_DIR = "debug_sent"
os.makedirs(DEBUG_DIR, exist_ok=True)

def _get_preprompt(path):
    """Внутренняя функция для загрузки системной инструкции."""
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except:
            return ""
    return ""

def _log_debug(prompt, response):
    """Запись истории запросов для отладки."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        with open(os.path.join(DEBUG_DIR, f"ai_req_{ts}.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)
        with open(os.path.join(DEBUG_DIR, f"ai_res_{ts}.txt"), "w", encoding="utf-8") as f:
            f.write(response)
    except Exception as e:
        print(f"[AI-ОШИБКА] Не удалось записать лог: {e}")

def process_ai_request(text, config):
    """
    Основная точка входа для обработки текста нейросетью.
    Возвращает текст ответа, если сработала активация, иначе None.
    """
    activation_word = config.get("ai_activation_phrase", "").lower()
    
    # Проверяем, нужно ли вообще обращаться к ИИ
    if not activation_word or activation_word not in text.lower():
        return None

    print(f"[AI] Фраза активации найдена. Подготовка запроса к {config['llm_model']}...")
    
    preprompt = _get_preprompt(config.get("ai_preprompt_path", "preprompt.txt"))
    full_prompt = f"{preprompt}\n\nUser: {text}" if preprompt else text
    
    try:
        payload = {
            "model": config["llm_model"],
            "prompt": full_prompt,
            "stream": False
        }
        
        response = requests.post(
            config["llm_url"], 
            json=payload, 
            timeout=60
        )
        
        if response.status_code == 200:
            ai_response = response.json().get('response', '').strip()
            _log_debug(full_prompt, ai_response)
            return ai_response
        else:
            print(f"[AI-ОШИБКА] Сервер Ollama вернул код {response.status_code}")
            return f"Ошибка ИИ: {response.status_code}"
            
    except Exception as e:
        print(f"[AI-ОШИБКА] Ошибка при обращении к LLM: {e}")
        return None