# One image for both services (api and ui): Python 3.12 + the runtime and UI packages + the code.
# The models are NOT in the image: Ollama runs on the Mac and is reached via host.docker.internal.
FROM python:3.12-slim
WORKDIR /app
# Requirements first, code second: editing code then reuses the cached pip layer (fast rebuilds).
COPY requirements.txt requirements-ui.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ui.txt
COPY rag_minimal.py api.py ui.py ./
COPY copilot ./copilot
COPY docs ./docs
EXPOSE 8000 8501
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
