FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the API port
EXPOSE 8000

# Start uvicorn — auto-migrations run on startup via create_tables()
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
