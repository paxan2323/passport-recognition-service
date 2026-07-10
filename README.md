# Passport Recognition Service

HTTP-сервис и CLI для распознавания российских паспортов на базе [docreader-ocr](https://pypi.org/project/docreader-ocr/) + постобработка (`postprocess/`).

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Модели (~200 МБ) скачиваются автоматически при первом запуске в `~/.cache/docreader/models/`.

## Использование

```bash
python main.py path/to/document.jpg        # финальные поля после постобработки
python main.py --raw path/to/document.jpg  # + сырой OCR для сравнения
```

Или из кода:

```python
from postprocess import build_reader, refine_passport, MrzOcr

reader = build_reader()          # DocReader + ретраи низкоконфидентных зон
result = reader.process("passport.jpg", return_crops=True)
fields = refine_passport(result.documents[0], MrzOcr())
print({k: f.as_dict() for k, f in fields.items()})
```

## Постобработка (пакет `postprocess`)

Поверх сырого OCR docreader-ocr:

- **`retry_ocr`** — зоны с низким confidence перечитываются после
  препроцессинга (апскейл, CLAHE, подавление бликов, бинаризация);
- **`mrz`** — парсер MRZ внутреннего паспорта (ICAO 9303): контрольные
  цифры подтверждают серию/номер, дату рождения, дату выдачи и код
  подразделения; ФИО восстанавливаются из транслитерации;
- **`mrz_ocr`** — MRZ перечитывается латинской моделью EasyOCR с
  allowlist (кириллическая сеть docreader читает её плохо); детектор
  паспорта MRZ не размечает, поэтому полоса берётся снизу кропа;
- **`normalize`** — форматная валидация полей (даты, серия/номер
  10 цифр, код `XXX-XXX`, ФИО — только кириллица);
- **`division_codes`** — «Кем выдан» подставляется из справочника
  [hflabs/fms-unit](https://github.com/hflabs/fms-unit) по коду
  подразделения;
- **`passport`** — арбитраж: числовые поля берутся из MRZ при
  сошедшейся чексумме, ФИО сверяются с MRZ, расхождения помечаются.

Тесты:

```bash
.venv\Scripts\python.exe -X utf8 tests\test_postprocess.py  # юнит, быстрые
.venv\Scripts\python.exe -X utf8 tests\test_mrz_e2e.py      # MRZ сквозной
```

## Passport Recognition Service

HTTP API for Russian passport OCR (docreader-ocr + postprocess). Images stay in RAM only (Redis TTL, no disk writes).

### Quick start (Docker)

```bash
# Set API keys: key:client_id pairs, comma-separated
set PASSPORT_API_KEYS=your-secret-key:my-client

docker compose up --build
```

Services:
- **nginx** — http://localhost:8080 (rate limit, 10MB body)
- **api** — http://localhost:8000 (direct)
- **Prometheus** — http://localhost:9090
- **Grafana** — http://localhost:3000 (admin/admin)

### Recognize (sync)

```bash
curl -X POST "http://localhost:8080/v1/passport/recognize" \
  -H "X-API-Key: your-secret-key" \
  -F "image=@passport.jpg"
```

Async (202 + poll):

```bash
curl -X POST "http://localhost:8080/v1/passport/recognize/async" \
  -H "X-API-Key: your-secret-key" \
  -F "image=@passport.jpg" \
  -F "webhook_url=https://example.com/hook"

curl -H "X-API-Key: your-secret-key" \
  "http://localhost:8080/v1/tasks/{task_id}"
```

Health: `GET /health`, `GET /ready`, `GET /metrics`, `GET /v1/limits`

### Web-панель (ручная загрузка + статистика)

После `docker compose up` откройте:

**http://localhost:8080/dashboard**

- загрузка паспорта drag-and-drop;
- таблица всех полей с confidence, source, confirmed;
- live-статистика: очередь, воркеры, запросы, отказы (обновление каждые 5 с);
- API Key сохраняется в браузере (localStorage).

JSON статистики: `GET /api/dashboard/stats`

Отключить панель: `DASHBOARD_ENABLED=false`

### Свои модели в Docker

По умолчанию worker в Docker качал **стоковые** веса в tmpfs (дообученная сеть терялась
при каждом рестарте). Чтобы использовать локальные модели из `~/.cache/docreader/models/`:

1. Скопируй `.env.example` → `.env` и проверь путь `DOCREADER_MODELS_HOST`
2. В каталоге должны быть:
   - `easyocr/model/custom_example.pth` — дообученная распознавалка
   - `regpage_obb.pt` — детектор прописки
   - `passport.pt`, `doc_classifier.pt`, … (остальные YOLO)
3. Перезапуск:

```bash
docker compose up -d --force-recreate worker nginx
```

Worker подхватит `RegistrationDetectorReader` (детектор прописки) и новую CRNN-сеть
без пересборки образа. Код `postprocess/` и `service/` монтируется volume'ом.

### Local dev (without Docker)

```bash
pip install -r requirements-service.txt
# Terminal 1 — Redis
redis-server --save "" --appendonly no
# Terminal 2 — worker (loads OCR models ~3s)
set PASSPORT_API_KEYS=dev-key-001:default
set REDIS_URL=redis://localhost:6379/0
python -m service.worker.main
# Terminal 3 — API
uvicorn service.api.main:app --reload --port 8000
```

### Tests

```bash
.venv\Scripts\python.exe -X utf8 tests\test_postprocess.py
.venv\Scripts\python.exe -m pytest tests/test_service.py tests/test_service_lifecycle.py -v
```

Lifecycle test needs local Redis on port 6379.

### Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `PASSPORT_API_KEYS` | `dev-key-001:default` | `key:client_id` pairs |
| `REDIS_URL` | `redis://redis:6379/0` | Queue + blob store |
| `QUEUE_MAX_DEPTH` | `50` | Backpressure threshold |
| `RATE_LIMIT_RPS` | `5` | Token bucket per key |
| `DASHBOARD_ENABLED` | `true` | Web UI at `/dashboard` |
| `WORKER_METRICS_URL` | `http://worker:9091/metrics` | Worker Prometheus scrape |
| `DOCREADER_MODELS_HOST` | `~/.cache/docreader/models` | Host mount for trained weights |
| `REGPAGE_WEIGHTS_PATH` | auto / `regpage_obb.pt` | Registration page detector |

See `service/core/config.py` for full list.

## Deployment (production)

Server path: `/opt/passport-recognition-service`.

- **HTTPS API:** `https://pass.hermes-support.ru`
- **Dashboard:** `https://pass.hermes-support.ru/dashboard`
- **Models:** `/var/lib/docreader/models` on host (custom CRNN + regpage detector)

```bash
# One-time server bootstrap (Docker + clone)
curl -fsSL https://raw.githubusercontent.com/paxan2323/passport-recognition-service/main/scripts/bootstrap-server.sh | bash

# Set API keys on server
nano /opt/passport-recognition-service/.env

# SSL (needs A-record pass.hermes-support.ru -> server IP)
bash /opt/passport-recognition-service/scripts/setup-ssl.sh

# Manual deploy
/opt/passport-recognition-service/scripts/deploy.sh
```

Auto-deploy: push to `main` runs GitHub Actions (tests → SSH deploy → health check).

Required GitHub secrets: `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`.

## Документация

- [GitHub](https://github.com/mishanyacorleone/docreader)
- [PyPI](https://pypi.org/project/docreader-ocr/)
