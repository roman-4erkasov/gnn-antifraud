# GNN Anti-Fraud Research Pipeline

Исследовательский пайплайн для применения Graph Neural Networks (GNN) к задачам антифрода. Проект организован как серия self-contained фаз, каждая со своей средой зависимостей, кодом, тестами и учебными материалами.

## Структура проекта

```
gnn-antifraud/
├── phases/                          # Все фазы исследования
│   ├── phase01_lightgbm_baseline/   # LightGBM baseline
│   ├── phase02_minimal_gnn/         # Minimal GCN
│   ├── phase03_structural_encoding/ # Structural features
│   ├── phase04a_xai_basics/         # XAI for fraud (methods)
│   ├── phase05_pattern_detection/   # Pattern detection
│   ├── phase05b_xai_validation/     # XAI validation on patterns
│   ├── phase06_temporal_graphs/     # Temporal GNNs
│   ├── phase07_cross_dataset/       # Cross-dataset evaluation
│   ├── phase08_scaling_sampling/    # Scaling & sampling
│   ├── phase09_distributed_inference/ # Distributed inference (PySpark)
│   ├── phase10_distributed_gnn_frameworks/ # Framework comparison
│   └── phase11_temporal_link_prediction/ # Temporal link prediction
├── data/                            # Общие датасеты (создаётся при загрузке)
├── src/utils/                       # Общие утилиты
│   ├── calibration.py               # Калибровка вероятностей
│   └── metrics.py                   # Метрики качества
└── README.md                        # Этот файл
```

## Фазы исследования

### Phase 01: LightGBM Baseline
**Зависимости:** нет  
**Описание:** Baseline модель на LightGBM с графовыми фичами.  
**Результат:** Рабочий baseline, 6 уроков, 3 ноутбука.

### Phase 02: Minimal GNN
**Зависимости:** Phase 01  
**Описание:** Минимальная реализация GCN на PyTorch Geometric.  
**Результат:** GCN модель, 5 уроков, 2 ноутбука.

### Phase 03: Structural Encoding
**Зависимости:** Phase 01, Phase 02  
**Описание:** Лапласианские собственные вектора, random walk features, Louvain communities.  
**Результат:** Structural features, 5 уроков, 2 ноутбука.

### Phase 04a: XAI Basics
**Зависимости:** Phase 01, Phase 02, Phase 03  
**Описание:** GNNExplainer, PGExplainer, GAT attention для объяснения предсказаний; валидация по chargeback.  
**Результат:** XAI методы, 6 уроков, 3 ноутбука.

### Phase 05: Pattern Detection
**Зависимости:** Phase 01, Phase 02, Phase 03  
**Описание:** Синтетические паттерны (Mitme, Cascade, Money Mule), graph-классификаторы, pattern templates.  
**Результат:** Pattern templates, 5 уроков, 2 ноутбука.

### Phase 05b: XAI Validation
**Зависимости:** Phase 04a, Phase 05  
**Описание:** Валидация XAI-объяснений на синтетических паттернах и найденных классификатором инстансах (precision/recall).  
**Результат:** XAI validation pipeline, 5 уроков, 2 ноутбука.

### Phase 06: Temporal Graphs
**Зависимости:** Phase 01, Phase 02, Phase 04a  
**Описание:** TGN, GRN, rolling GCN для временных графов.  
**Результат:** Temporal GNNs, 6 уроков, 3 ноутбука.

### Phase 07: Cross-Dataset Evaluation
**Зависимости:** Phase 01, Phase 02  
**Описание:** Transfer learning, domain adaptation, сравнение датасетов.  
**Результат:** Cross-dataset pipeline, 5 уроков, 2 ноутбука.

### Phase 08: Scaling & Sampling
**Зависимости:** Phase 02, Phase 06  
**Описание:** Mini-batch training, negative sampling, scaling на большие графы.  
**Результат:** SamplingWrapper для масштабирования, 6 уроков, 3 ноутбука.

### Phase 09: Distributed Inference
**Зависимости:** Phase 02-08 (soft dependency — использует обученные модели)  
**Описание:** Distributed GNN inference на PySpark с executor-based sampling.  
**Требования:** Docker для локального Spark кластера (1 master + 2 workers).  
**Результат:** Distributed inference pipeline, 6 уроков, 3 ноутбука.

### Phase 10: Distributed GNN Frameworks
**Зависимости:** Phase 09  
**Описание:** Сравнение DGL Distributed, AliGraph, Quiver.  
**Результат:** Benchmark suite, рекомендации, 5 уроков, 2 ноутбука.

### Phase 11: Temporal Link Prediction
**Зависимости:** нет (self-contained; использует только общие `src/utils/` и `data/`)  
**Описание:** Ранжирование новых связей в пределах горизонта прогноза `H`. Общий интерфейс датасетов (synthetic, sx-mathoverflow, tgbl-coin), GNN-эмбеддинги без утечки, pluggable retrieval (embedding/structural/I2I/sequential), целевой CatBoost listwise-ранкер.  
**Результат:** Link-prediction pipeline, 7 уроков, 3 ноутбука.

## Граф выполнения

```
Phase 01 (LightGBM)
    ↓
Phase 02 (Minimal GNN)
    ↓
Phase 03 (Structural Encoding)
    ↓
Phase 04a (XAI Basics) ───────┐
    ↓                         ↓
Phase 05 (Patterns)       Phase 06 (Temporal)
    ↓                         ↓
Phase 05b (XAI Validation)    ↓
    └─────────────────────────┴──→ Phase 07 (Cross-Dataset)
                                    ↓
                               Phase 08 (Scaling)
                                    ↓
                               Phase 09 (Distributed Inference)
                                    ↓
                               Phase 10 (Framework Comparison)

Phase 11 (Temporal Link Prediction) — self-contained, независима от фаз 01-10
```

## Начало работы

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) для управления зависимостями
- Docker (для Phase 09)

### Установка

Каждая фаза имеет собственную среду зависимостей:

```bash
# Перейти в директорию фазы
cd phases/phase01_lightgbm_baseline

# Установить зависимости
uv sync

# Запустить
uv run python run_lightgbm_baseline.py

# Запустить тесты
uv run pytest tests/
```

### Данные

Датасеты хранятся в общей директории `data/`. Каждая фаза имеет свой `data_loader.py`, который:
1. Проверяет наличие данных в `data/`
2. Скачивает/распаковывает если отсутствуют
3. Загружает данные для конкретной фазы

### Docker (для Phase 09)

```bash
cd phases/phase09_distributed_inference

# Поднять Spark кластер
docker-compose up -d

# Проверить статус
docker-compose ps

# Остановить
docker-compose down
```

## Учебные материалы

Каждая фаза содержит:
- **`lessons/`** — Markdown уроки с 5-частной структурой:
  1. **Explain** — теория
  2. **Code** — рабочий пример
  3. **Visualize** — графики и визуализации
  4. **Practice** — упражнение
  5. **Solution** — решение упражнения

- **`notebooks/`** — Интерактивные Jupyter ноутбуки для экспериментов

## Архитектурные решения

- **Self-contained phases:** Каждая фаза изолирована со своими зависимостями
- **Shared data:** Датасеты в общей директории `data/`
- **Shared utilities:** Общие утилиты в `src/utils/`
- **Wrapper/Adapter:** Phase 08 использует wrapper для масштабирования моделей из других фаз без их модификации
- **Docker for distributed:** Phase 09 использует Docker Compose для локального Spark кластера

## Дополнительные ресурсы

- [OpenSpec changes](openspec/changes/) — Детальные specs для каждой фазы
- [Lessons](phases/*/lessons/) — Учебные материалы
- [Notebooks](phases/*/notebooks/) — Интерактивные ноутбуки

## Лицензия

MIT

## Контакты

Для вопросов и предложений создайте issue в репозитории.
