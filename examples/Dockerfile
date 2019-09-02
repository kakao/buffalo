FROM ubuntu:18.04
MAINTAINER recoteam <recoteam@kakaocorp.com>

RUN apt-get update && apt-get install -y python3.7 python3-pip git cmake wget locales &&\
    pip3 install virtualenv

RUN mkdir -p /home/toros
WORKDIR /home/toros

RUN /bin/bash -c "virtualenv venv --python=python3 && source ./venv/bin/activate &&\
    pip install numpy cython && pip install n2"

RUN /bin/bash -c "source ./venv/bin/activate && git clone -b master https://github.com/kakao/buffalo.git buffalo.git &&\
    cd buffalo.git && git submodule update --init && python setup.py install && pip install -r requirements.txt"
