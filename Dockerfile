FROM python:3.9-slim

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt

ENTRYPOINT [ "bash" ]