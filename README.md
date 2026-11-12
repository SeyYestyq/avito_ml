# Авито ML — Ранжирование рекомендаций

Соревнование по ранжированию объявлений в рекомендательной ленте Авито.  
**Метрика:** NDCG · **Призы:** 🥇 750K · 🥈 450K · 🥉 300K

## Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone https://github.com/SeyYestyq/avito_ml.git
cd avito_ml
```

### 2. Скачать данные

Данные (~18 GB) нужно скачать с платформы соревнования и положить в корень проекта:

```
avito_ml/
├── competition.zip          ← скачать с платформы
├── sample_submission.parquet ← скачать с платформы (или из архива)
├── baselines/
├── run.sh
├── GUIDE.md
└── ...
```

### 3. Запустить

```bash
chmod +x run.sh
./run.sh
```

Скрипт автоматически:
- Распакует `competition.zip`
- Создаст Python-окружение (нужен [uv](https://docs.astral.sh/uv/))
- Пропатчит CatBoost на CPU (если нет CUDA)
- Обучит CatBoost YetiRank + Transformer
- Сгенерирует `submission.parquet` для загрузки

### Пошаговый запуск (без скрипта)

```bash
unzip competition.zip
cd baselines
uv sync --locked --no-config --default-index https://pypi.org/simple
source .venv/bin/activate

# CatBoost (для CPU: поменять task_type в скрипте)
python catboost_baseline.py train --data ../competition --output ../runs/catboost-01
python catboost_baseline.py predict --data ../competition \
    --model ../runs/catboost-01/model.cbm --output ../runs/catboost-01/submission.parquet

# Transformer
python neural_baseline.py --device cpu train --data ../competition --output ../runs/neural-01
python neural_baseline.py --device cpu predict --data ../competition \
    --model ../runs/neural-01/model.pt --output ../runs/neural-01/submission.parquet
```

## Требования

- Python 3.11 (uv установит автоматически)
- ~50 GB свободного места
- 16+ GB RAM (при нехватке — ограничить `train_feeds` в конфигах)
- GPU не обязательна (всё работает на CPU)

## Подробный гайд

См. [GUIDE.md](GUIDE.md) — описание данных, признаков, моделей и идеи для улучшения.
