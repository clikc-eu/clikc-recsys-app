FROM python:3.8

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN apt-get update \
    && apt-get install gcc -y \
    && apt-get clean

RUN pip install -r /app/requirements.txt \
    && python -m spacy download it_core_news_lg \
    && rm -rf /root/.cache/pip

COPY . /app/