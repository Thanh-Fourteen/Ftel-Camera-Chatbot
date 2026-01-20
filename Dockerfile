# ---------- Stage 1: builder ----------
FROM python:3.11-alpine AS builder

WORKDIR /install

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---------- Stage 2: runtime ----------
FROM python:3.11-alpine

WORKDIR /app

COPY --from=builder /install /usr/local
COPY src ./src

ENV PYTHONUNBUFFERED=1

EXPOSE 8686

CMD ["uvicorn", "api.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8686"]
