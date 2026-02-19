Здесь хранится входной CSV с агрегатами активности команды.

Ожидаемые поля:
- team_id (string)
- period_start (YYYY-MM-DD)
- meetings_count (int)
- meetings_minutes (float)
- after_hours_ratio (float, 0..1)
- commits_count (int)
- active_days (int)
- tasks_completed (int)
- tasks_reopened (int)
- messages_count (int)
- context_switches (int)
- deep_work_minutes (float)
- burnout_label (0/1) — только для обучения

Если файл `team_activity.csv` отсутствует, training‑сервис сгенерирует
синтетический датасет автоматически.
Генератор делает дисбаланс класса (~15% положительных),
добавляет сезонность, корреляции и нулевые дни активности.

При генерации также создаются разбиения в `data/processed/`:
`train.csv`, `valid.csv`, `test.csv`.

Минимальный пример для ручных тестов:
- `sample_team_activity.csv`
Чтобы использовать его для обучения, установите
`DATA_PATH=data/raw/sample_team_activity.csv` в `.env`.
