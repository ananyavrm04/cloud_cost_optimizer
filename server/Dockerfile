FROM python:3.11-slim
WORKDIR /app

# Install dependencies once (openenv-core already included in requirements.txt)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only runtime files needed by the environment server
COPY server /app/server
COPY tasks /app/tasks
COPY models.py /app/models.py
COPY client.py /app/client.py
COPY __init__.py /app/__init__.py
COPY openenv.yaml /app/openenv.yaml

# Set PYTHONPATH so absolute imports resolve from /app
ENV PYTHONPATH=/app

# HF Spaces uses port 7860
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
