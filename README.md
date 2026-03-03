# Система классификации новостей с ИИ

## Описание проекта

Это ML-сервис для интеллектуальной классификации новостных текстов (BBC News) на категории `business`, `entertainment`, `politics`, `sport`, `tech`.
Проект разделен на два независимых сервиса:
- `Inference` — FastAPI-сервис для онлайн-предсказаний (работает 24/7)
- `Training` — сервис обучения по требованию (one-shot запуск)

Система использует общий preprocessing для train/inference, внутренний реестр моделей и конфигурацию через `.env`.
## Основные возможности

### 🤖 Инференс и предсказания
- **Онлайн-классификация текста**: endpoint `/predict` возвращает класс, confidence и вероятности по классам
- **Горячая смена модели**: endpoint `/reload` загружает новую версию из реестра без остановки сервиса
- **Контроль состояния**: `/health`, `/stats`, `/registry` для мониторинга

### 📊 Обучение и валидация
- **Обучение по запросу**: training-сервис запускается отдельно и завершается после выполнения
- **Стратифицированный split**: train/valid/test через `sklearn.model_selection.train_test_split`
- **Валидация качества до деплоя**: модель проходит пороги по `f1_macro` и `accuracy`

### 🗂️ Управление моделями
- **Внутренний реестр моделей**: версии хранятся в `model_registry/<version>/`
- **Метрики и метаданные**: сохраняются вместе с каждой моделью
- **Автоподгрузка в inference**: training отправляет сигнал на `/reload`

## Архитектура системы

### 🏗️ Основные компоненты

**Inference сервис** (`src/inference/`)
- FastAPI-приложение и REST API
- Загрузка модели из реестра на старте
- Предсказания и перезагрузка версии модели

**Training сервис** (`src/training/`)
- Загрузка данных из CSV
- Подготовка данных и обучение моделей
- Проверка метрик и деплой в реестр

**Общий слой** (`src/common/`)
- `config.py` — конфигурация из `.env`
- `preprocessing.py` — общий preprocessing текста
- `dataset.py` — стратифицированное разбиение
- `registry.py` — файловый реестр моделей
- `logging.py` — настройка логирования

## Технологический стек

### 🐍 Backend
- **Python 3.9+**
- **FastAPI**
- **Pydantic**

### 🤖 Машинное обучение
- **scikit-learn**
- **pandas / numpy**

### 🔧 Инструменты разработки
- **uv** — менеджер зависимостей
- **pytest** — тестирование
- **Docker / Docker Compose** — контейнеризация и запуск

## Структура проекта

```text
ML_service_task/
  data/
    raw/                   # исходные данные (в т.ч. bbc-text.csv)
    processed/             # служебные/подготовленные артефакты
  model_registry/          # хранилище версий моделей
  src/
    common/                # общий код
    inference/             # FastAPI сервис предсказаний
    training/              # пайплайн обучения
  tests/                   # unit-тесты
  Dockerfile.inference
  Dockerfile.training
  docker-compose.yml
  pyproject.toml
```

## Конфигурация

### ⚙️ `.env` параметры

```env
DATA_PATH=data/raw/bbc-text.csv
PROCESSED_DIR=data/processed
REGISTRY_PATH=model_registry
MODEL_VERSION=latest

INFERENCE_HOST=0.0.0.0
INFERENCE_PORT=8000
INFERENCE_RELOAD_URL=http://inference:8000/reload

METRICS_MIN_F1_MACRO=0.90
METRICS_MIN_ACCURACY=0.90

TRAIN_RATIO=0.7
VALID_RATIO=0.2
TEST_RATIO=0.1

RANDOM_SEED=42
TEXT_USE_STOPWORDS=true
TEXT_USE_STEM=false
TEXT_USE_LEMMA=false
MAX_FEATURES=50000
LOG_LEVEL=INFO
```

## Использование

### 🚀 Запуск через Docker Compose

1. Запуск inference:
```bash
docker compose up --build inference
```

2. Запуск training по требованию:
```bash
docker compose run --rm training
```

3. Запуск на порту `8001`:
```env
INFERENCE_PORT=8001
```

### 🚀 Локальный запуск (Windows, `.venv`, Python 3.9)

1. Подготовка окружения и зависимостей через `uv`:
```bash
py -3.9 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip uv
.\.venv\Scripts\uv pip compile pyproject.toml -o requirements.txt
.\.venv\Scripts\uv pip install -r requirements.txt
.\.venv\Scripts\uv pip install pytest
```

2. Обучение:
```bash
$env:PYTHONPATH="src"; .\.venv\Scripts\python -m training.train
```

3. Запуск inference:
```bash
$env:PYTHONPATH="src"; .\.venv\Scripts\python -m uvicorn inference.app:app --host 127.0.0.1 --port 8001
```

## 📡 API эндпоинты

- `GET /health` — проверка состояния сервиса
- `POST /predict` — предсказание категории текста
- `POST /reload` — перезагрузка модели (latest/конкретная версия)
- `GET /registry` — список версий моделей и метрик
- `GET /stats` — статистика запросов

Пример запроса:

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"text": "Government plans new tax reforms in parliament."},
      {"text": "Team wins league match after extra time."}
    ]
  }'
```

## 🧪 Тестирование

```bash
.\.venv\Scripts\python -m pytest tests
```
