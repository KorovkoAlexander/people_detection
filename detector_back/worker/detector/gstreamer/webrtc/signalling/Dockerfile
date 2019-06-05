FROM ubuntu:disco

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update \
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

# webrtc dependencies:
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        gstreamer1.0-nice libgirepository1.0-dev gir1.2-gst-plugins-bad-1.0

RUN pip3 install --user websockets netifaces

WORKDIR /opt/
COPY . /opt/

EXPOSE 8443/tcp

CMD ["python3", "-u", "./signalling.py"]
