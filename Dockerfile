FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir . && useradd --create-home appuser && chown -R appuser:appuser /app
COPY docker-entrypoint.sh /usr/local/bin/failurelens-entrypoint
RUN chmod +x /usr/local/bin/failurelens-entrypoint
USER appuser
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["failurelens-entrypoint"]
EXPOSE 8000
CMD ["uvicorn", "failurelens.api:app", "--host", "0.0.0.0", "--port", "8000"]
HEALTHCHECK --interval=10s --timeout=3s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')"

