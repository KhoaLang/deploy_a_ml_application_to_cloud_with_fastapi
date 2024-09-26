#!/bin/bash

# start nginx server to map port 5000 to 80
service nginx start

# start app
gunicorn -k uvicorn.workers.UvicornWorker --reload main:app