FROM nvidia/cuda:10.1-runtime-ubuntu18.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        git \
        libglib2.0-0 \
        libjpeg8-dev \
        libsm6 \
        libxext6 \
        libxrender1 \
        make \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-venv \
        python3-wheel \
        python3.6 \
        zlib1g-dev \
    && apt-get clean


ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /worker

COPY ./detector /worker/detector

RUN pip3 install detector/

COPY . /worker

RUN pip3 install .

CMD ["python3", "-u", "worker.py"]