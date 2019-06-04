FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        python3.6


WORKDIR /opt/rambler-ai/ssds.pytorch
COPY resource/ sota-benchmark/resource/

COPY setup.* .
RUN pip3 install -e .
COPY detector detector

COPY sota-benchmark sota-benchmark
WORKDIR sota-benchmark

CMD ["python3", "-u", "run.py"]
