#!/usr/bin/env bash
# ============================================================================
# Авито: Ранжирование рекомендаций — полный скрипт запуска
# ============================================================================
# Машина: MacBook M4, 16 GB RAM, без CUDA → обучаем на CPU
# ============================================================================

set -euo pipefail

PROJECT_DIR="/Users/damir/Documents/хакатоны/Авито"
BASELINES_DIR="${PROJECT_DIR}/baselines"
DATA_DIR="${PROJECT_DIR}/competition"

echo "========================================="
echo "  Шаг 0: Создание директории для результатов"
echo "========================================="
mkdir -p "${PROJECT_DIR}/runs"

# ─────────────────────────────────────────────
# ШАГ 1: Распаковать данные (18 GB → ~40+ GB)
# ─────────────────────────────────────────────
echo "========================================="
echo "  Шаг 1: Распаковка competition.zip"
echo "========================================="
if [ ! -d "${DATA_DIR}" ]; then
    echo "Распаковка competition.zip (~18 GB)..."
    echo "⚠️  Убедитесь, что на диске есть минимум 50 GB свободного места!"
    cd "${PROJECT_DIR}"
    unzip -o competition.zip
    echo "✅ Данные распакованы в ${DATA_DIR}"
else
    echo "✅ Данные уже распакованы"
fi

# Проверка структуры
echo ""
echo "Проверка структуры данных..."
for split in train val test; do
    count=$(ls "${DATA_DIR}/${split}/impressions/"*.parquet 2>/dev/null | wc -l | tr -d ' ')
    emb_size=$(ls -lh "${DATA_DIR}/${split}/item_embeddings/data.memmap" 2>/dev/null | awk '{print $5}')
    echo "  ${split}: ${count} parquet-файлов, embeddings: ${emb_size}"
done
echo ""

# ─────────────────────────────────────────────
# ШАГ 2: Создать Python-окружение
# ─────────────────────────────────────────────
echo "========================================="
echo "  Шаг 2: Установка Python-окружения через uv"
echo "========================================="
cd "${BASELINES_DIR}"

if [ ! -d ".venv" ]; then
    echo "Создание виртуального окружения..."
    # pyproject.toml требует Python 3.11.x
    uv sync --locked --no-config --default-index https://pypi.org/simple
    echo "✅ Окружение создано"
else
    echo "✅ Окружение уже существует"
fi

# Активируем окружение для дальнейших команд
source .venv/bin/activate
echo "Python: $(python --version)"
echo ""

# ─────────────────────────────────────────────
# ШАГ 3: Обучение CatBoost (CPU-режим)
# ─────────────────────────────────────────────
echo "========================================="
echo "  Шаг 3: Обучение CatBoost baseline"
echo "========================================="
echo ""
echo "⚠️  CatBoost baseline по умолчанию настроен на GPU."
echo "    У тебя Mac M4 без CUDA — нужно переключить на CPU."
echo "    Скрипт автоматически пропатчит catboost_baseline.py"
echo ""

# Патчим CatBoost для работы на CPU (если ещё не пропатчен)
if grep -q '"task_type": "GPU"' catboost_baseline.py; then
    echo "Переключаю CatBoost на CPU..."
    sed -i.bak \
        -e 's/"task_type": "GPU"/"task_type": "CPU"/' \
        -e '/"gpu_ram_part"/d' \
        -e '/"devices": "0"/d' \
        catboost_baseline.py
    echo "✅ CatBoost переключён на CPU"
fi

RUN_CB="${PROJECT_DIR}/runs/catboost-01"
if [ ! -d "${RUN_CB}" ]; then
    echo "Запуск обучения CatBoost..."
    echo "Ожидаемое время: 30-90 минут на CPU (16 GB RAM)"
    echo ""
    python catboost_baseline.py train \
        --data "${DATA_DIR}" \
        --output "${RUN_CB}"
    echo ""
    echo "✅ CatBoost обучен! Модель: ${RUN_CB}/model.cbm"
    echo "   Метрики: ${RUN_CB}/run.json"
else
    echo "✅ CatBoost уже обучен в ${RUN_CB}"
fi
echo ""

# ─────────────────────────────────────────────
# ШАГ 4: Предсказания CatBoost на test
# ─────────────────────────────────────────────
echo "========================================="
echo "  Шаг 4: Предсказания CatBoost на test"
echo "========================================="
SUB_CB="${RUN_CB}/submission.parquet"
if [ ! -f "${SUB_CB}" ]; then
    python catboost_baseline.py predict \
        --data "${DATA_DIR}" \
        --model "${RUN_CB}/model.cbm" \
        --output "${SUB_CB}"
    echo "✅ Предсказания CatBoost: ${SUB_CB}"
else
    echo "✅ Предсказания CatBoost уже готовы: ${SUB_CB}"
fi
echo ""

# ─────────────────────────────────────────────
# ШАГ 5: Обучение Transformer (CPU-режим)
# ─────────────────────────────────────────────
echo "========================================="
echo "  Шаг 5: Обучение Transformer baseline"
echo "========================================="
echo "Ожидаемое время: 2-6 часов на CPU (5 эпох)"
echo "Совет: на M4 можно попробовать --device mps, но baseline"
echo "       поддерживает только cpu и cuda."
echo ""

RUN_NN="${PROJECT_DIR}/runs/neural-01"
if [ ! -d "${RUN_NN}" ]; then
    python neural_baseline.py --device cpu train \
        --data "${DATA_DIR}" \
        --output "${RUN_NN}"
    echo ""
    echo "✅ Transformer обучен! Модель: ${RUN_NN}/model.pt"
else
    echo "✅ Transformer уже обучен в ${RUN_NN}"
fi
echo ""

# ─────────────────────────────────────────────
# ШАГ 6: Предсказания Transformer на test
# ─────────────────────────────────────────────
echo "========================================="
echo "  Шаг 6: Предсказания Transformer на test"
echo "========================================="
SUB_NN="${RUN_NN}/submission.parquet"
if [ ! -f "${SUB_NN}" ]; then
    python neural_baseline.py --device cpu predict \
        --data "${DATA_DIR}" \
        --model "${RUN_NN}/model.pt" \
        --output "${SUB_NN}"
    echo "✅ Предсказания Transformer: ${SUB_NN}"
else
    echo "✅ Предсказания Transformer уже готовы: ${SUB_NN}"
fi
echo ""

# ─────────────────────────────────────────────
# ШАГ 7: Итоговая сводка
# ─────────────────────────────────────────────
echo "========================================="
echo "  🎉 Готово!"
echo "========================================="
echo ""
echo "📁 Файлы для загрузки на платформу:"
echo "   CatBoost:    ${SUB_CB}"
echo "   Transformer: ${SUB_NN}"
echo ""
echo "📊 Метрики обучения:"
if [ -f "${RUN_CB}/run.json" ]; then
    echo "   CatBoost (run.json):"
    python -c "
import json
with open('${RUN_CB}/run.json') as f:
    d = json.load(f)
m = d.get('metrics', {})
for k, v in m.items():
    print(f'     {k}: {v}')
"
fi
echo ""
if [ -f "${RUN_NN}/run.json" ]; then
    echo "   Transformer (run.json):"
    python -c "
import json
with open('${RUN_NN}/run.json') as f:
    d = json.load(f)
h = d.get('history', [])
if h:
    best = min(h, key=lambda x: x['val_logloss'])
    print(f'     Лучшая эпоха: {best[\"epoch\"]}')
    print(f'     Val logloss:  {best[\"val_logloss\"]:.6f}')
"
fi
echo ""
echo "💡 Подсказка: загрузи один из submission.parquet на платформу"
echo "   и проверь score на public лидерборде."
echo ""

