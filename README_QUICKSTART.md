# Zenith Investment Platform - Quick Start 🚀

## Pré-requisitos

- Docker & Docker Compose
- Python 3.11+ (apenas para desenvolvimento local)
- 8GB RAM mínimo

## Início Rápido

### 1. Build e Start dos Serviços

```bash
# Build das imagens
docker-compose build

# Iniciar todos os serviços
docker-compose up -d

# Verificar status
docker-compose ps
```

### 2. Verificar Logs

```bash
# Todos os serviços
docker-compose logs -f

# Apenas backend
docker-compose logs -f backend

# Apenas celery worker
docker-compose logs -f celery-worker

# Apenas celery beat (tarefas agendadas)
docker-compose logs -f celery-beat
```

### 3. Treinar Modelos Iniciais

```bash
# Entrar no container do backend
docker-compose exec backend bash

# Executar script de treinamento
python scripts/train_initial_models.py
```

Ou via API diretamente:

```bash
# Treinar PETR4.SA
curl -X POST "http://localhost:8000/api/v1/ml/train" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA", "epochs": 100}'
```

### 4. Testar a API

```bash
# Health check
curl http://localhost:8000/health

# Documentação interativa (Swagger)
open http://localhost:8000/docs

# Executar suite de testes
./backend/scripts/test_api.sh
```

### 5. Fazer Previsões

#### Previsão de Preço (LSTM)

```bash
curl -X POST "http://localhost:8000/api/v1/ml/predict/price" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "PETR4.SA",
    "horizon": 5,
    "use_cache": true
  }' | jq '.'
```

#### Sinal de Trading (Ensemble)

```bash
curl -X POST "http://localhost:8000/api/v1/ml/predict/signal" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "PETR4.SA",
    "use_cache": true
  }' | jq '.'
```

#### Previsão Comprehensiva

```bash
curl -X POST "http://localhost:8000/api/v1/ml/predict/comprehensive" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "PETR4.SA",
    "horizon": 5,
    "use_cache": true
  }' | jq '.'
```

## Estrutura de Serviços

```
┌──────────────────────────────────────────────────┐
│                   Docker Stack                   │
├──────────────────────────────────────────────────┤
│                                                  │
│  🚀 Backend (FastAPI)         :8000              │
│  └─ API REST + ML Endpoints                      │
│                                                  │
│  🐘 PostgreSQL                :5432              │
│  └─ Banco de dados principal                     │
│                                                  │
│  📦 Redis                     :6379              │
│  └─ Cache de predições                           │
│                                                  │
│  🔄 Celery Worker                                │
│  └─ Fila: ml_training, maintenance               │
│                                                  │
│  ⏰ Celery Beat                                  │
│  └─ Tarefas agendadas (retraining)               │
│                                                  │
└──────────────────────────────────────────────────┘
```

## Tarefas Agendadas (Celery Beat)

- **Semanal** (Domingos, 2h): Retreinamento de modelos
- **Mensal** (Dia 1, 3h): Limpeza de versões antigas
- **Diário** (Meia-noite): Limpeza de cache

## Comandos Úteis

### Gerenciar Serviços

```bash
# Parar todos os serviços
docker-compose stop

# Reiniciar um serviço
docker-compose restart backend

# Ver logs em tempo real
docker-compose logs -f --tail=100

# Remover todos os containers
docker-compose down

# Remover incluindo volumes
docker-compose down -v
```

### Acesso aos Containers

```bash
# Backend
docker-compose exec backend bash

# PostgreSQL
docker-compose exec postgres psql -U user -d db

# Redis CLI
docker-compose exec redis redis-cli
```

### Banco de Dados

```bash
# Migrations (Alembic)
docker-compose exec backend alembic upgrade head

# Criar nova migration
docker-compose exec backend alembic revision --autogenerate -m "description"
```

### Testes

```bash
# Executar testes
docker-compose exec backend pytest tests/ -v

# Com coverage
docker-compose exec backend pytest tests/ --cov=src/ml --cov-report=html

# Testes específicos
docker-compose exec backend pytest tests/test_ml_predictor.py -v
```

## Monitoramento

### Cache Redis

```bash
# Estatísticas do cache
curl http://localhost:8000/api/v1/ml/cache/stats

# Invalidar cache de um ticker
curl -X DELETE http://localhost:8000/api/v1/ml/cache/PETR4.SA

# Limpar todo o cache
curl -X DELETE http://localhost:8000/api/v1/ml/cache
```

### Modelos

```bash
# Listar modelos disponíveis
curl http://localhost:8000/api/v1/ml/models

# Filtrar por ticker
curl "http://localhost:8000/api/v1/ml/models?ticker=PETR4.SA"

# Filtrar por tipo
curl "http://localhost:8000/api/v1/ml/models?model_type=lstm"
```

### Celery

```bash
# Ver workers ativos
docker-compose exec celery-worker celery -A src.core.celery_app inspect active

# Ver tarefas agendadas
docker-compose exec celery-beat celery -A src.core.celery_app inspect scheduled

# Ver estatísticas
docker-compose exec celery-worker celery -A src.core.celery_app inspect stats
```

## Troubleshooting

### Container não inicia

```bash
# Ver logs detalhados
docker-compose logs backend

# Rebuildar sem cache
docker-compose build --no-cache backend

# Verificar configuração
docker-compose config
```

### Erro de conexão com PostgreSQL

```bash
# Verificar se PostgreSQL está rodando
docker-compose ps postgres

# Testar conexão
docker-compose exec backend python -c "from sqlalchemy import create_engine; engine = create_engine('$DATABASE_URL'); print(engine.connect())"
```

### Erro de conexão com Redis

```bash
# Verificar se Redis está rodando
docker-compose ps redis

# Testar conexão
docker-compose exec redis redis-cli ping
```

### Modelos não encontrados

```bash
# Verificar diretório de modelos
docker-compose exec backend ls -la /app/models/

# Treinar modelos novamente
python scripts/train_initial_models.py
```

## Variáveis de Ambiente

Arquivo `.env`:

```env
# Database
DATABASE_URL=postgresql+asyncpg://user:password@postgres:5432/db

# Redis
REDIS_URL=redis://redis:6379/0

# ML
MODELS_PATH=/app/models
ML_CACHE_TTL=3600

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:8000

# Celery
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
```

## Documentação

- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **ML Module**: `backend/src/ml/README.md`
- **Implementation Summary**: `backend/ML_IMPLEMENTATION_SUMMARY.md`

## Suporte

- Issues: https://github.com/Zenith-Investment/zenith-platform/issues
- Docs: `backend/src/ml/README.md`

## ⚠️ Disclaimer

Todas as previsões são apenas para fins educacionais e NÃO constituem recomendação de investimento.
