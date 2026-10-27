FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1

# CPU-only PyTorch keeps the image small; inference here is a few ms per request
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY serving/requirements.txt serving/requirements.txt
RUN pip install --no-cache-dir -r serving/requirements.txt

COPY *.py ./
COPY serving ./serving
COPY tests ./tests

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
