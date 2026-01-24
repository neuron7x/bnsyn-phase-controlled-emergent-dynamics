FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY benchmarks /app/benchmarks
COPY docs /app/docs
COPY tests /app/tests
COPY scripts /app/scripts

RUN pip install --no-cache-dir -e ".[dev]"

CMD ["python", "-m", "pytest", "-m", "not validation"]
