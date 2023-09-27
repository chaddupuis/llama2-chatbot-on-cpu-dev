FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

RUN apt update -y
RUN apt install -y python3 python3-pip

COPY ./requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install -r requirements.txt
COPY . /opt/app



