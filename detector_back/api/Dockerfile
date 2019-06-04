FROM python:3.6-slim

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /api

COPY . /api

RUN pip3 install .

CMD ["python", "-u", "api.py"]