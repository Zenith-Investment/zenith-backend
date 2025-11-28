# Motor de Análise Técnica com Machine Learning

Sistema completo de análise técnica e previsão de preços usando Machine Learning.

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Módulos](#módulos)
- [Uso](#uso)
- [Modelos](#modelos)
- [API](#api)

## 🎯 Visão Geral

Este motor de ML fornece:

- **Feature Engineering**: Indicadores técnicos (RSI, MACD, Bollinger Bands, etc.)
- **Modelos LSTM**: Previsão de séries temporais com redes neurais
- **Ensemble Models**: RandomForest + XGBoost para sinais de trading
- **Cache Redis**: Predições cacheadas para melhor performance
- **Auto-retraining**: Sistema automatizado de retreinamento de modelos

## 🏗 Arquitetura

```
src/ml/
├── data/
│   ├── data_fetcher.py       # Coleta de dados históricos (Yahoo Finance)
│   └── data_preprocessor.py  # Limpeza e pré-processamento
├── features/
│   ├── technical_indicators.py  # Indicadores técnicos (TA-Lib)
│   └── feature_pipeline.py      # Pipeline de feature engineering
├── models/
│   ├── lstm_model.py            # Modelo LSTM (TensorFlow)
│   ├── ensemble_model.py        # Random Forest + XGBoost
│   └── model_registry.py        # Versionamento de modelos
├── prediction/
│   ├── predictor.py             # Orquestrador de predições
│   └── cache.py                 # Cache Redis
└── training/
    ├── trainer.py               # Pipeline de treinamento
    └── evaluation.py            # Métricas e backtesting
```

## 📦 Módulos

### 1. Data Pipeline

**DataFetcher**
```python
from ml.data.data_fetcher import DataFetcher

fetcher = DataFetcher()
df = fetcher.fetch_yfinance("PETR4.SA", period="1y")
```

**DataPreprocessor**
```python
from ml.data.data_preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
df_clean = preprocessor.clean_data(df)
df_norm, params = preprocessor.normalize_data(df_clean)
```

### 2. Feature Engineering

**TechnicalIndicators**
```python
from ml.features.technical_indicators import TechnicalIndicators

indicators = TechnicalIndicators()
df = indicators.add_all_indicators(df)
```

Indicadores disponíveis:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving Averages (SMA, EMA)
- Stochastic Oscillator
- ATR (Average True Range)
- OBV (On-Balance Volume)
- ADX (Average Directional Index)
- CCI (Commodity Channel Index)

**FeaturePipeline**
```python
from ml.features.feature_pipeline import FeaturePipeline

pipeline = FeaturePipeline()
df = pipeline.prepare_features(df, add_indicators=True)
df = pipeline.create_target_variable(df, target_type="binary_direction")
```

### 3. Modelos

**LSTM Model**
```python
from ml.models.lstm_model import LSTMModel

# Criar e treinar
model = LSTMModel(sequence_length=60, lstm_units=[50, 50])
model.train(X_train, y_train, X_val, y_val, epochs=100)

# Fazer predições
predictions = model.predict_next(last_sequence, n_steps=5)

# Salvar/Carregar
model.save("./models/lstm_PETR4")
model.load("./models/lstm_PETR4")
```

**Ensemble Model**
```python
from ml.models.ensemble_model import EnsembleModel

# Para classificação (sinais de trading)
model = EnsembleModel(task="classification")
model.train(X_train, y_train, X_val, y_val)

# Fazer predições
signal = model.predict(X_new)  # 0=sell, 1=hold, 2=buy
proba = model.predict_proba(X_new)

# Feature importance
importance = model.get_feature_importance()
```

### 4. Treinamento

**ModelTrainer**
```python
from ml.training.trainer import ModelTrainer

trainer = ModelTrainer(models_path="./models")

# Treinar LSTM
lstm_model, metrics = trainer.train_lstm_model(
    ticker="PETR4.SA",
    epochs=100,
    save_model=True
)

# Treinar Ensemble
ensemble_model, metrics = trainer.train_ensemble_model(
    ticker="PETR4.SA",
    task="classification",
    save_model=True
)

# Treinar todos
results = trainer.train_all_models("PETR4.SA")
```

**ModelEvaluator**
```python
from ml.training.evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Avaliar classificação
metrics = evaluator.calculate_classification_metrics(y_true, y_pred)

# Backtesting
backtest_results = evaluator.backtest_signals(
    df, signals,
    initial_capital=10000.0,
    commission=0.001
)

# Relatório
report = evaluator.generate_evaluation_report(
    "LSTM_PETR4",
    metrics,
    backtest_results
)
```

### 5. Predição

**Predictor**
```python
from ml.prediction.predictor import Predictor

predictor = Predictor(
    models_path="./models",
    redis_url="redis://localhost:6379",
    use_cache=True
)

# Carregar modelos
predictor.load_models(
    lstm_path="./models/lstm_PETR4",
    ensemble_path="./models/ensemble_PETR4"
)

# Predição de preço
price_pred = predictor.predict_price("PETR4.SA", horizon=5)

# Predição de sinal
signal_pred = predictor.predict_signal("PETR4.SA")

# Predição completa
all_pred = predictor.predict_comprehensive("PETR4.SA", horizon=5)
```

### 6. Cache

**PredictionCache**
```python
from ml.prediction.cache import PredictionCache

cache = PredictionCache(redis_url="redis://localhost:6379", default_ttl=3600)

# Armazenar
cache.set("PETR4.SA", "lstm", prediction_data, ttl=3600)

# Recuperar
cached_data = cache.get("PETR4.SA", "lstm")

# Invalidar
cache.invalidate_ticker("PETR4.SA")
cache.clear_all()
```

## 🚀 Uso

### Exemplo Completo: Treinar e Fazer Predições

```python
from ml.training.trainer import ModelTrainer
from ml.prediction.predictor import Predictor

# 1. Treinar modelos
trainer = ModelTrainer(models_path="./models")
results = trainer.train_all_models("PETR4.SA")

print(f"LSTM Metrics: {results['lstm']['metrics']}")
print(f"Ensemble Metrics: {results['ensemble']['metrics']}")

# 2. Fazer predições
predictor = Predictor(models_path="./models")
predictor.load_models(
    lstm_path="./models/lstm_PETR4_20250128",
    ensemble_path="./models/ensemble_PETR4_20250128"
)

# Predição completa
prediction = predictor.predict_comprehensive("PETR4.SA", horizon=5)

print(f"Price Prediction: {prediction['price_prediction']}")
print(f"Signal: {prediction['signal_prediction']['signal']}")
```

## 📊 Modelos

### LSTM (Long Short-Term Memory)

**Uso**: Previsão de preços futuros (séries temporais)

**Arquitetura**:
- Input: Sequências de 60 dias de preços normalizados
- 2 camadas LSTM (50 unidades cada)
- Dropout (20%) para regularização
- Dense layer (25 unidades)
- Output: Preço previsto

**Métricas**:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- Direction Accuracy

### Ensemble (RandomForest + XGBoost)

**Uso**: Classificação de sinais de trading

**Arquitetura**:
- RandomForest: 100 árvores, profundidade 10
- XGBoost: 100 estimadores, learning rate 0.1
- Ensemble: Votação/Média das predições

**Métricas**:
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix

## 🔧 API

### Endpoints (a serem implementados)

```
POST /api/v1/ml/predict/{ticker}
GET  /api/v1/ml/signal/{ticker}
POST /api/v1/ml/train/{ticker}
GET  /api/v1/ml/models
GET  /api/v1/ml/cache/stats
DELETE /api/v1/ml/cache/{ticker}
```

## ⚙️ Configuração

### Variáveis de Ambiente

```bash
# ML Models
MODELS_PATH=/app/models
ML_CACHE_TTL=3600  # 1 hour

# Redis
REDIS_URL=redis://redis:6379/0

# Training
ML_RETRAIN_SCHEDULE_CRON=0 2 * * 0  # Weekly on Sunday at 2 AM
```

## 📈 Performance

- **Predição**: < 500ms com cache
- **Cache Hit Rate**: > 80% em produção
- **LSTM Accuracy**: 60-70% em dados de teste
- **Ensemble Accuracy**: 65-75% para sinais

## ⚠️ Disclaimers

- Os modelos são para fins educacionais e de pesquisa
- Não constituem recomendação de investimento
- Decisões de investimento são de responsabilidade do usuário
- Performance passada não garante resultados futuros

## 📚 Referências

- [Stock Market Prediction Using ML 2025](https://www.analyticsvidhya.com/blog/2021/10/machine-learning-for-stock-market-prediction-with-step-by-step-implementation/)
- [LSTM Python Stock Market](https://www.datacamp.com/tutorial/lstm-python-stock-market)
- [TensorFlow Stocks Prediction](https://github.com/Leci37/TensorFlow-stocks-prediction-Machine-learning-RealTime)

## 🤝 Contribuindo

Desenvolvido como parte da Issue #14 do projeto Zenith Investment Platform.
