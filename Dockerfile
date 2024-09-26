FROM python:3.8-slim

USER root

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# g++ nginx\
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && apt-get remove --purge -y gnupg lsb-release && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . /app
WORKDIR /app

RUN pip install -r /app/requirements.txt

# RUN cp /app/nginx/nginx.conf /etc/nginx/nginx.conf
# RUN cp /app/nginx/nginx.conf /etc/nginx/conf.d/virtual.conf

EXPOSE 80

# CMD ["bash", "-c", "/app/start.sh"]
# CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--reload", "main:app"]
CMD ["fastapi", "run", "app/main.py", "--port", "80"]