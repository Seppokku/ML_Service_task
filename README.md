# Система прогнозирования выгорания команды с помощью ML

## Описание проекта

Это ML‑сервис для прогнозирования риска выгорания по анонимным паттернам
активности (митинги, коммиты, сообщения, глубинная работа). Проект объединяет
два независимых сервиса: **Inference** (24/7, предсказания и UI) и **Training**
(обучение по требованию). Модели хранятся в файловом регистре версий.

Подробная архитектура и связи между файлами — в `PROJECT.md`.

## Основные возможности

### 🤖 Интеллектуальный прогноз выгорания
- **Оценка риска**: модель возвращает `risk_score` и флаг `is_high_risk`
- **Горячая замена модели**: `/reload` подхватывает новую версию без даунтайма
- **Decision threshold**: порог решения хранится в метаданных модели

### 📊 Автоматическая обработка данных
- **Общий препроцессинг**: единые фичи для train и inference
- **Регистр моделей**: версии, метрики, метаданные
- **Синтетика**: генератор данных, если CSV отсутствует

### 💬 UI‑интерфейс и мониторинг
- **Статус и метрики**: health, статистика, PR‑AUC/Recall
- **Переобучение**: кнопка + загрузка CSV
- **Quick Predict**: отправка запроса прямо из UI
- **Светлая/тёмная тема**

## Архитектура системы

### 🏗️ Основные компоненты

**Inference сервис** (`src/inference/`)
- FastAPI endpoints: `/predict`, `/health`, `/reload`, `/registry`, `/stats`, `/train`
- UI Dashboard: `src/inference/static/index.html`
- ModelService: загрузка и инференс модели

**Training сервис** (`src/training/`)
- CatBoost + подбор гиперпараметров
- Метрики: PR‑AUC, Recall@P, best F1/F2
- Валидация и запись в регистр

**Общий слой** (`src/common/`)
- Конфиги `.env`
- Препроцессинг и derived‑фичи
- Файловый регистр моделей
- Генерация синтетических данных

**Данные и модели**
- `data/raw/` — CSV данные
- `model_registry/` — версии моделей

**Контейнеры**
- `Dockerfile.inference`, `Dockerfile.training`, `docker-compose.yml`

## Технологический стек

### 🐍 Backend
- **Python 3.9+**
- **FastAPI** + **Uvicorn**
- **Pydantic**

### 🤖 Машинное обучение
- **CatBoost**
- **scikit‑learn**
- **pandas / numpy**

### 🗄️ Хранилище
- файловый регистр моделей (`model_registry/`)

### 🔧 Инструменты разработки
- **uv** — менеджер зависимостей
- **pytest** — тестирование
- **Docker / Docker Compose**

## Структура данных

### 📋 Основные сущности

**TeamActivity (CSV row)**
- Фичи активности + `burnout_label`

**ModelVersion**
- `model.joblib`, `metrics.json`, `metadata.json`

**TrainingMetrics**
- PR‑AUC, Recall@P, best F1/F2, threshold

## Использование

### 🚀 Запуск системы (Docker)

1. Запуск inference:
```bash
docker compose up --build
```

2. Обучение (по требованию):
```bash
docker compose run --rm training
```

UI:
```
http://localhost:8000/
```

### 🚀 Локальный запуск (через uv)

1. Создать окружение и установить зависимости:
```bash
uv venv .venv
uv pip compile pyproject.toml -o requirements.txt
uv pip install -r requirements.txt
```

2. Обучение:
```bash
PYTHONPATH=src python -m training.train
```

3. Запуск inference:
```bash
PYTHONPATH=src uvicorn inference.app:app --host 127.0.0.1 --port 8000
```

## 📡 API эндпоинты

- `GET /health` — состояние сервиса
- `GET /stats` — статистика запросов
- `GET /registry` — список моделей
- `POST /predict` — предсказание
- `POST /reload` — смена версии модели
- `POST /train` — переобучение (можно CSV)
- `GET /` или `/ui` — UI Dashboard

## Как пользоваться

1) Запусти сервис (Docker или локально).  
2) Проверь `/health`.  
3) Сделай запрос в `/predict`.  
4) При необходимости — переобучи через `/train` или UI.  

## Пример запроса /predict
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "team_id": "team-1",
        "period_start": "2026-02-01",
        "meetings_count": 12,
        "meetings_minutes": 420,
        "after_hours_ratio": 0.35,
        "commits_count": 26,
        "active_days": 5,
        "tasks_completed": 18,
        "tasks_reopened": 4,
        "messages_count": 160,
        "context_switches": 22,
        "deep_work_minutes": 240
      }
    ]
  }'
```

## Формат CSV

```
team_id, period_start,
meetings_count, meetings_minutes, after_hours_ratio,
commits_count, active_days, tasks_completed, tasks_reopened,
messages_count, context_switches, deep_work_minutes,
burnout_label
```

Минимальный пример: `data/raw/sample_team_activity.csv`.

## Конфигурация (.env)

Ключевые параметры:
```
DATA_PATH=data/raw/team_activity.csv
REGISTRY_PATH=model_registry
MODEL_VERSION=latest

METRICS_MIN_PR_AUC=0.3
METRICS_MIN_RECALL=0.05
RECALL_PRECISION_THRESHOLD=0.6
```

Параметры генерации синтетики:
```
LABEL_POS_RATE=0.18
LABEL_NOISE_STD=0.12
LABEL_SHARPNESS=2.0
LABEL_SIGNAL_SCALE=1.15
```

## 🧪 Тестирование
```bash
pytest tests/
```
