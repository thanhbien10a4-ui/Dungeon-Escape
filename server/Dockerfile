FROM python:3.12-slim
WORKDIR /app/server
COPY server/pyproject.toml ./
COPY server/app ./app
COPY web /app/web
RUN pip install --no-cache-dir .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
