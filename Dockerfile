# Stage 1: Builder
FROM python:3.11.8-slim AS builder

WORKDIR /app

# Copy dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11.8-slim AS runtime

WORKDIR /app

# Copy installed Python packages and executables
COPY --from=builder /install /usr/local

# Copy app source
COPY . .

# Create non-root user
RUN useradd -m appuser
USER appuser

# Start the app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
