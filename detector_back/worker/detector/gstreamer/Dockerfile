FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# We need the latest gstreamer which fixes some bugs which are critical for us,
# hence we switch from Ubuntu Bionic to the currently latest Disco.
RUN sed -i 's/bionic/disco/g' /etc/apt/sources.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        gstreamer1.0-libav \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-python3-plugin-loader \
        gstreamer1.0-tools \
        gstreamer1.0-x \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        python3-dev \
        python3-gst-1.0 \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        python3.7 \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer1.0-dev \
        libmsgpack-dev \
        libmsgpackc2 \
        libopencv-dev \
        libtool \
        pkg-config


WORKDIR /opt/rambler-ai/ssds.pytorch
COPY resource/ gstreamer/resource/

ENV LD_LIBRARY_PATH = gstreamer/resource/TensorRT-5.1.2.2/lib
RUN pip install gstreamer/resource/TensorRT-5.1.2.2/python/tensorrt-5.1.2.2-cp37-none-linux_x86_64.whl

COPY setup.* .
RUN pip3 install -e .
COPY detector detector

COPY gstreamer gstreamer

RUN cd gstreamer/gst-plugin \
        && ./autogen.sh \
        && make clean \
        && make \
        && make install

WORKDIR gstreamer
