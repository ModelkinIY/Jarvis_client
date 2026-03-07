# Jarvis_client

curl http://localhost:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_fixed.wav" \
  -F "model=large-v3-turbo" \
  -F "language=ru" \
  -F "response_format=json"

time curl http://localhost:8000/v1/audio/transcriptions -F "file=@test_fixed.wav" -F "model=large-v3-turbo"