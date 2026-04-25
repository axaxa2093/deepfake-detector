# Deepfake Detector API

Микросервис на FastAPI для детекции дипфейков на основе модели Dual-Stream + Attention (EfficientNetB0 + FFT + Cross-Attention).

**Метрики:** Accuracy 92.6% · AUC-ROC 98.2% · F1 92.4%

---

## Структура проекта

```
deepfake_service/
├── main.py
├── requirements.txt
├── Dockerfile
├── best_model_attention.keras
├── normalization_params.txt
└── README.md
```

---

## Запуск

### Локально
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Docker
```bash
docker build -t deepfake-detector .
docker run -p 8000:8000 deepfake-detector
```

---

## Эндпоинты

### `GET /health`
```json
{"status": "OK"}
```

### `POST /predict`

| Поле    | Тип  | Описание              |
|---------|------|-----------------------|
| `image` | файл | Фото лица (jpg / png) |

```json
{
  "label": "real",
  "fake_probability": 0.0209,
  "real_probability": 0.9791,
  "threshold": 0.5
}
```

---

## Примеры запросов

**curl**
```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -F "image=@face.jpg"
```

**Python**
```python
import requests

resp = requests.post(
    "http://localhost:8000/predict",
    files={"image": open("face.jpg", "rb")},
)
print(resp.json())
```

---

## Swagger

```
http://localhost:8000/docs
```
