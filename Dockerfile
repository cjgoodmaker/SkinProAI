# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
FROM python:3.10-slim

# HF Spaces runs as uid 1000
RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install curl first, then use it to add the NodeSource repo, then install Node.js
RUN apt-get update && \
    apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY --chown=user requirements.txt ml-requirements.txt
COPY --chown=user backend/requirements.txt api-requirements.txt
RUN pip install --no-cache-dir -r ml-requirements.txt -r api-requirements.txt

# Build React frontend
COPY --chown=user web/ web/
WORKDIR /app/web
RUN npm ci && npm run build

WORKDIR /app

# Copy application source
COPY --chown=user models/ models/
COPY --chown=user backend/ backend/
COPY --chown=user mcp_server/ mcp_server/
COPY --chown=user data/case_store.py data/case_store.py
COPY --chown=user guidelines/ guidelines/

# Pre-download MONET and sentence-transformers models so first request is fast
RUN python -c "\
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification; \
AutoProcessor.from_pretrained('chanwkim/monet'); \
AutoModelForZeroShotImageClassification.from_pretrained('chanwkim/monet'); \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Models cached')"

# Runtime data directories — must be writable by user 1000
RUN mkdir -p data/uploads data/patient_chats data/lesions && \
    echo '{"patients": []}' > data/patients.json && \
    chown -R user:user data/

USER user

# Fix OMP_NUM_THREADS warning and ensure all CPU cores are used
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
