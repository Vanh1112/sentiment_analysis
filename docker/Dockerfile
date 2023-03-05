FROM python:3.6-slim

COPY requirements.txt /app/
WORKDIR /app
RUN pip3 install --no-cache-dir -r requirements.txt

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8