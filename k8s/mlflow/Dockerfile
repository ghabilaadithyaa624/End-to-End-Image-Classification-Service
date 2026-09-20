FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    mlflow==3.16.1 \
    boto3 \
    psycopg2-binary

COPY start.sh /app/start.sh

RUN chmod +x /app/start.sh

EXPOSE 5000

CMD ["/app/start.sh"]
