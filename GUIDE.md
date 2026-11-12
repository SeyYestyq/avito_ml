# 🏆 Авито: Ранжирование рекомендаций — Полный гайд

## Суть задачи

**Ранжирование объявлений** в рекомендательной ленте Авито. Для каждой ленты (`feed_id`) дан набор кандидатов-объявлений. Нужно присвоить каждому кандидату `score` — чем выше, тем выше объявление в ленте. Метрика — **NDCG** (средний по лентам).

**Призы:** 🥇 750 000₽ · 🥈 450 000₽ · 🥉 300 000₽

---

## 📂 Структура данных

```
competition/
├── sample_submission.parquet      ← шаблон ответа (feed_id, item_id, score=0)
├── train/
│   ├── impressions/part-*.parquet ← ленты с признаками + target
│   └── item_embeddings/
│       ├── data.memmap            ← матрица int8, (N, 128)
│       └── data.meta              ← JSON с dtype, shape, order
├── val/                           ← аналогично train, есть target
└── test/                          ← аналогично, НЕТ target
```

### Колонки в impressions (одна строка = одна лента)

| Колонка | Тип | Описание |
|---|---|---|
| `feed_id` | int32 | Уникальный ID ленты |
| `timestamp` | int32 | Unix-время записи |
| `user_id` | int32 | Обезличенный ID пользователя |
| `category_id` | list\<int8\> | Категория каждого кандидата (M штук) |
| `f1`–`f5` | list\<float32\> | 5 нормализованных числовых признаков (M штук) |
| `pos` | list\<int16\> | Исходная позиция (только train) |
| `target` | list\<int8\> | Бинарная метка: 1 = целевое действие (только train/val) |
| `item_id__embedding_idx` | list\<int32\> | Индекс строки в data.memmap; −1 = нет эмбеддинга |
| `item_id` | list\<int32\> | ID объявления |
| `user_embedding` | list\<int8\> | 128-мерный вектор пользователя |
| `history__item_id` | list\<int32\> | До 256 ID товаров из истории (новые → старые) |
| `history__category_id` | list\<int8\> | Категории товаров из истории |
| `history__event_type` | list\<int8\> | Тип события |
| `history__event_source` | list\<int8\> | Источник события |
| `history__item_id__embedding_idx` | list\<int32\> | Индексы эмбеддингов товаров из истории |

> [!IMPORTANT]
> Все массивы длины M (кандидаты) согласованы по индексу: `item_id[i]`, `f1[i]`, `target[i]` — про одно объявление. При развороте в строки — разворачивать все списки вместе.

> [!WARNING]
> `target` — метка для обучения, **не входной признак**. Нельзя использовать при инференсе.

### Эмбеддинги

- **data.memmap** — бинарная матрица `int8`, shape `(N, 128)`, C-order
- Перед вычислениями (dot, cosine) **обязательно** приводить к `float32`
- Индекс `-1` = эмбеддинга нет → подставлять нулевой вектор

### Временные рамки

| Часть | Период | Назначение |
|---|---|---|
| Train | 6 – 19 сентября 2026 | Обучение |
| Val | 22 – 24 сентября 2026 | Подбор гиперпараметров |
| Test | 26 – 29 сентября 2026 | Сабмит (Public 30% / Private 70%) |

---

## 🚀 Запуск — пошагово

### Предусловия

- ✅ macOS, Apple M4, 16 GB RAM
- ✅ `uv` установлен (`/opt/homebrew/bin/uv`)
- ❌ CUDA нет → всё на CPU
- ⚠️ Нужно ~50 GB свободного места (18 GB zip + ~35 GB распакованных данных)

### Вариант 1: Одной командой

```bash
cd /Users/damir/Documents/хакатоны/Авито
./run.sh
```

Скрипт [run.sh](file:///Users/damir/Documents/хакатоны/Авито/run.sh) автоматически:
1. Распакует `competition.zip`
2. Создаст Python 3.11 окружение через `uv`
3. Пропатчит CatBoost на CPU (вместо GPU)
4. Обучит CatBoost
5. Сделает предсказания CatBoost
6. Обучит Transformer
7. Сделает предсказания Transformer

### Вариант 2: Пошагово (рекомендуется для контроля)

```bash
# 1. Распаковка данных
cd /Users/damir/Documents/хакатоны/Авито
unzip -o competition.zip

# 2. Окружение
cd baselines
uv sync --locked --no-config --default-index https://pypi.org/simple
source .venv/bin/activate

# 3. Обучение CatBoost (сначала пропатчить на CPU!)
#    Замените в catboost_baseline.py:
#      "task_type": "GPU"  →  "task_type": "CPU"
#    и удалите строки с "gpu_ram_part" и "devices"
python catboost_baseline.py train \
    --data ../competition --output ../runs/catboost-01

# 4. Предсказания CatBoost
python catboost_baseline.py predict \
    --data ../competition --model ../runs/catboost-01/model.cbm \
    --output ../runs/catboost-01/submission.parquet

# 5. Обучение Transformer
python neural_baseline.py --device cpu train \
    --data ../competition --output ../runs/neural-01

# 6. Предсказания Transformer
python neural_baseline.py --device cpu predict \
    --data ../competition --model ../runs/neural-01/model.pt \
    --output ../runs/neural-01/submission.parquet
```

---

## 🧠 Что обучать — два бейзлайна

### 1. CatBoost Ranker (YetiRank)

**Что делает:** Градиентный бустинг для ранжирования. Каждая лента — одна группа, внутри которой CatBoost оптимизирует порядок.

**Признаки (9 штук):**

| # | Признак | Откуда |
|---|---|---|
| 1–5 | `f1`–`f5` | Числовые, уже нормализованы |
| 6 | `category_id` | Категориальный |
| 7 | `embedding_dot` | Скалярное произведение user_emb · item_emb |
| 8 | `embedding_cosine` | Косинусное сходство user_emb и item_emb |
| 9 | `embedding_missing` | Флаг: нет ли эмбеддинга у кандидата |

**Гиперпараметры:**
- YetiRank, до 2000 деревьев, lr = 0.4, L2 = 5
- Early stopping по NDCG@30 на val (50 итераций без улучшения)
- `use_best_model = True`

**Время обучения:** ~30–90 минут на CPU

**Выход:** `model.cbm` + `run.json` с метриками

### 2. Transformer

**Что делает:** Candidate-set Transformer. Токен пользователя + все кандидаты ленты обрабатываются через self-attention.

**Архитектура:**
```
User embedding (128) → Linear → 64
Candidate: [f1–f5, item_emb_norm(128), missing_flag, cat_emb(8)] → Linear → 64
[user_token, cand_1, ..., cand_M] → TransformerEncoder (2 слоя, 4 головы) → Linear → 1
```

~85K параметров. Loss = Binary Cross-Entropy.

**Гиперпараметры:**
- 5 эпох, batch = 32 ленты, AdamW, lr = 0.001, weight_decay = 1e-4
- Лучший checkpoint по val logloss

**Время обучения:** ~2–6 часов на CPU

**Выход:** `model.pt` + `run.json`

---

## 📊 Что можно улучшить (идеи для победы)

> [!TIP]
> Бейзлайны **не используют** историю пользователя, позицию кандидата и user_id/item_id. Это огромный резерв.

### Фичи, которые стоит добавить

1. **История пользователя** (до 256 событий):
   - Пересечение категорий истории и кандидатов
   - Косинусная близость item-эмбеддингов кандидата с эмбеддингами из истории (средняя, максимальная, top-k)
   - Число событий по каждому `event_type` и `event_source`
   - Был ли кандидат в истории (item_id match)
   - TF-IDF / BM25 по category_id истории vs category_id кандидата

2. **Позиция** (`pos`, только train):
   - Position bias features
   - Обучение с position debiasing (IPW)

3. **Агрегаты по пользователю/товару:**
   - Средние target по user_id, item_id, category_id из train
   - CTR-like фичи (с учётом утечки — использовать только train!)

4. **Эмбеддинги:**
   - PCA/SVD эмбеддингов → дополнительные фичи
   - L2-норма как признак "уверенности"
   - Пересечения эмбеддингов истории с кандидатами (attention-pooling)

5. **Временные фичи:**
   - День недели, час из `timestamp`
   - Давность событий в истории

### Модельные улучшения

- **Ансамбль** CatBoost + Transformer (взвешенная сумма или стэкинг)
- **LambdaRank / ListNet** вместо YetiRank
- Увеличить число деревьев / слоёв трансформера
- **Cross-validation** по времени
- **MPS-ускорение** для PyTorch на M4 (потребуется допатчить `neural_baseline.py`)

---

## 📤 Формат сабмита

Parquet-файл с тремя колонками:

| Колонка | Тип | Описание |
|---|---|---|
| `feed_id` | int32 | ID тестовой ленты |
| `item_id` | int32 | ID объявления |
| `score` | float32 | Оценка: больше = выше в ленте |

Шаблон: `sample_submission.parquet` (или `sample_submission (1).parquet`). Все пары из шаблона должны быть в ответе, без лишних, без дубликатов, без NaN.

---

## ⏱ Оценка времени

| Этап | Время на M4 CPU |
|---|---|
| Распаковка 18 GB zip | 5–15 минут |
| `uv sync` (окружение) | 2–5 минут |
| CatBoost train | 30–90 минут |
| CatBoost predict | 5–15 минут |
| Transformer train (5 эпох) | 2–6 часов |
| Transformer predict | 10–30 минут |
| **Итого** | **~3–8 часов** |

> [!CAUTION]
> На 16 GB RAM с полным датасетом возможны проблемы с памятью. Если возникнет OOM:
> - В `catboost_baseline.py` поставьте `DATA_PARAMS = {"train_feeds": 50000, "val_feeds": 10000}`
> - В `neural_baseline.py` поставьте `"train_feeds": 50000, "val_feeds": 10000` в `TRAIN_CONFIG`
> - Это обрежет данные, но позволит обучиться и получить первый сабмит.
