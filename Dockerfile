FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Run as an ordinary user, not root, so a bug in the service has less reach.
RUN useradd --create-home --uid 1000 app
COPY --chown=app:app . .
USER app

EXPOSE 8080

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]