FROM ubuntu:16.04

RUN apt -y update
RUN apt -y upgrade

RUN apt install -y software-properties-common
RUN apt install -y build-essential

RUN add-apt-repository -y ppa:ubuntugis/ppa

RUN apt install -y python3 python3-pip

RUN mkdir /home/qualcity
COPY . /home/qualcity
WORKDIR /home/qualcity
RUN pip3 install -r requirements.txt
RUN ./setup.py install
RUN ./tests/utils_test.py
