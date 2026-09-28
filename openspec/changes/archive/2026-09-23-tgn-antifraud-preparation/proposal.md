## Why

Перед началом работы над системой рекомендаций для AntiFraud аналитиков на графах с 300M узлами, необходимо эмпирически оценить:
- нужна ли GNN вообще для задачи (vs tabular baseline),
- как влияет структурное кодирование при sparse node features (~3 признака),
- какие современные сэмплеры дают лучший tradeoff quality/speed,
- насколько Explainable AI работает на практике для интерпретации предсказаний.

Без этих данных нельзя принять архитектурные решения для продакшн-системы.

## What Changes

- Набор экспериментальных пайплайнов для 2 типов fraud: Credit Card Fraud (IEEE-CIS) и Mitme-Fraud (EEV), каждый в версии PyG и DGL для прямого сравнения фреймворков
- Сравнительная таблица: LightGBM (no graph) vs GCN/GAT vs TGN с structural encoding (ReCWE/RWSE)
- Бенчмарк графовых сэмплеров: Neighbor Sampling, GraphSAINT, GraphBolt — speed vs quality
- XAI-эксперименты: GNNExplainer, PGExplainer, attention analysis — качество интерпретаций на known fraud patterns
- Synthetic data generator для контролируемого тестирования (known ground truth о структурах fraud)
- CPU-first design: все эксперименты отлаживаются на CPU, GPU-ускорение как опция

## Capabilities

### New Capabilities
- `tgn-antifraud-preparation`: Research framework для эмпирической оценки эффективности GNN, structural encoding, сэмплеров и XAI на антифрод датасетах с дублирующей реализацией на PyG и DGL

### Modified Capabilities
<!-- None -->

## Impact

- Новый код в `src/`: experiments, datasets, models, samplers, explainers
- Зависимости: PyTorch, PyG, DGL, LightGBM, PySpark (для feature engineering)
- Data layer: download/caching of open datasets, synthetic data generation
- No breaking changes to existing codebase
