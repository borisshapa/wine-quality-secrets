FROM python:3.10-slim

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt

HEALTHCHECK CMD timeout 10s bash -c ':> /dev/tcp/127.0.0.1/80' || exit 1

#ENTRYPOINT ["bash"]