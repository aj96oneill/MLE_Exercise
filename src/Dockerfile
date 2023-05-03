# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /src

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

CMD ["gunicorn", "wsgi:app" , "--workers", "4", "--bind", "0.0.0.0:1313"]