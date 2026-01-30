FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY requirements-lock.txt pyproject.toml ./

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --require-hashes -r requirements-lock.txt && \
    pip install --no-cache-dir --no-deps -e .

COPY . .

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN chown -R appuser:appuser /workspace

USER appuser

RUN pytest -m "not validation" --cov=src/bnsyn --cov-fail-under=85 -q

ENTRYPOINT ["pytest", "-m", "not validation", "-v"]
CMD []
