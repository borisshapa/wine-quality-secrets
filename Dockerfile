FROM python:3.10-slim

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt /app

ADD . /app

RUN rm .cache/pip -rf

RUN pip install -r requirements.txt
