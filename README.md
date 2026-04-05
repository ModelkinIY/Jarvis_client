# Jarvis_client

curl http://localhost:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_fixed.wav" \
  -F "model=large-v3-turbo" \
  -F "language=ru" \
  -F "response_format=json"

time curl http://localhost:8000/v1/audio/transcriptions -F "file=@test_fixed.wav" -F "model=large-v3-turbo"


tts test
 curl -X POST "http://127.0.0.1:8001/generate"      -H "Content-Type: application/json"      -d '{"text": "Прив+ет, я Дж+арвис. Сист+ема с+интеза р+ечи гот+ова к раб+оте.", "speaker": "aidar"}'      --output test.wav

for make win .exe
pyinstaller --onefile --windowed `
    --name "JarvisAssistant" `
    --add-data "silero_vad.onnx;." `
    --collect-submodules onnxruntime `
    --hidden-import ai_module `
    client.py