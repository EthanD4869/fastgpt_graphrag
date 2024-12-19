FROM ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive

ENV LANG C.UTF-8

RUN apt-get update && apt-get install apt-utils python3 -y python3-pip -y docker.io -y vim -y libcudart11.0 libcublaslt11 -y ffmpeg -y opencc -y

RUN mkdir -p /root/.pip

ADD pip.conf /root/.pip/

ADD requirements.txt /root/requirements.txt

USER root

WORKDIR /root

RUN pip3 install -r requirements.txt

ADD api.py /root/api.py

CMD ["python3", "api.py"]
