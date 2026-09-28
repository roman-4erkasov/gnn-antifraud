## Why

Графовые модели (GNN) демонстрируют обещание в обнаружении мошенничества, но нет систематического эмпирического исследования, которое бы обосновало необходимость графовых представлений по сравнению с табличными baselines (LightGBM) на реальных сценариях антифрода. Существующее исследование было сфокусировано на TGN для временного антифрода, но практические препятствия (отсутствие открытых временных fraud-датасетов) сделали этот план нереализуемым. Необходимо переосмыслить подход: сверху вниз — от простого к сложному, с обязательной калибровкой, XAI и строгим сравнением across models, datasets and graph/non-graph representations.

## What Changes

- Создаётся полный исследовательский пайплайн из 8 фаз: от LightGBM baseline через минимальный GNN до временных графов, XAI, pattern detection, cross-dataset comparison и масштабирования.
- Обязательная калибровка (изотоническая регрессия) для всех вероятностных предсказаний и метрика Brier score.
- XAI (GNNExplainer / PGExplainer / Attention weights) для валидации модели и формирования следственных лидов.
- Фазы 1-5 фокусируются на fraud; фаза 6 — на временных графах в любом domain (снимает проблему отсутствия data).
- Фаза 7 — синтетическое сравнение across models и datasets, статистическая значимость, generalization.
- Фаза 8 — sampling optimization и стратегия масштабирования до 300M nodes.

## Capabilities

### New Capabilities

- `scoring`: Классификация транзакций на мошеннические. Включает: LightGBM baseline (3 признака), Graph-aware LightGBM, GCN, GAT, калибровку (изотоническая регрессия), метрики (PR-AUC, ROC-AUC, Brier score, log-loss, Precision@K).
- `xai`: Объяснение предсказаний моделей. Включает: GNNExplainer, PGExplainer, Attention weights, валидацию через synthetic patterns и domain labels, формирование investigation leads.
- `patterns`: Обнаружение паттернов мошенничества: Mitme fraud, Cascade fraud, Money mule detection. Сравнение XAI output с ground truth.
- `temporal`: Моделирование временных графов: TGN, GRN, rolling-window GCN. Валидация на temporal anomaly datasets (Pay-At-Pump, Sungkyunkwan, Naver Plus Bank).
- `sampling`: Оптимизация сэмплирования: Neighbor Sampling, GraphSAINT. Speed vs quality tradeoff. Partitioning strategy.

### Modified Capabilities

- (none)

## Impact

- Полностью новый исследовательский фреймворк в репозитории.
- Зависимости: PyTorch, PyTorch Geometric, LightGBM, scikit-learn, SHAP, graph-bolt (reference).
- src/ пока пуста — все артефакты создаются с нуля.
- Старое изменение `tgn-antifraud-preparation` устаревает и будет архивировано.
