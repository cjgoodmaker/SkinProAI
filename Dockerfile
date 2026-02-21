FROM python:3.10-slim

WORKDIR /app

# Install Node.js for building the React frontend
RUN apt-get update && \
    apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ml-requirements.txt
COPY backend/requirements.txt api-requirements.txt
RUN pip install --no-cache-dir -r ml-requirements.txt -r api-requirements.txt

# Build React frontend
COPY web/ web/
WORKDIR /app/web
RUN npm ci && npm run build

WORKDIR /app

# Copy application source
COPY models/ models/
COPY backend/ backend/
COPY data/case_store.py data/case_store.py
COPY guidelines/ guidelines/

# Runtime directories (writable by the app)
RUN mkdir -p data/uploads data/patient_chats data/lesions && \
    echo '{"patients": []}' > data/patients.json

# HF Spaces runs as a non-root user — ensure data dirs are writable
RUN chmod -R 777 data/

# HF Spaces uses port 7860
EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
