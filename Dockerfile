FROM python:3.11-slim
WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install openenv-core
RUN pip install "openenv-core[core]>=0.2.1"

# Copy all project files
COPY . .

# Set PYTHONPATH so absolute imports resolve from /app
ENV PYTHONPATH=/app

# HF Spaces uses port 7860
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
