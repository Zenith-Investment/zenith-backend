# InvestAI Platform - Backend

Backend API for the InvestAI Platform built with FastAPI.

## Setup

```bash
poetry install
poetry run alembic upgrade head
poetry run uvicorn src.main:app --reload
```
