FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY streamlit_app ./streamlit_app
COPY models ./models
COPY reports ./reports

RUN pip install --upgrade pip setuptools wheel \
    && pip install .

EXPOSE 8000

CMD ["uvicorn", "fraud_detection.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
