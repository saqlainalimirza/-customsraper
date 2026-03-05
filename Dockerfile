FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m playwright install --with-deps chromium

COPY . .

EXPOSE 8000

# Install bash for ulimit support
RUN apt-get update && apt-get install -y bash && rm -rf /var/lib/apt/lists/*

CMD ["bash", "-c", "ulimit -u 65535 && ulimit -n 65535 && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
