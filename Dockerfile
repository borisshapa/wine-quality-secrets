FROM python:3.10-slim

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt

HEALTHCHECK CMD curl --fail http://localhost:5000/ || exit 1

#ENTRYPOINT ["bash"]