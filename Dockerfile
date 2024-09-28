FROM python:3.8-slim

USER root

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && apt-get remove --purge -y gnupg lsb-release && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . /app
WORKDIR /app

RUN pip install -r /app/requirements.txt

RUN pip install -e .

CMD ["fastapi", "run", "/app/main.py", "--port", "10000"]
