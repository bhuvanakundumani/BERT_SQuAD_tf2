FROM tensorflow/tensorflow:2.5.0-gpu

RUN apt-get update && apt-get install -y python3-pip nano && /usr/bin/python3 -m pip install -U pip

ADD requirements.txt /tmp/

RUN /usr/bin/python3 -m pip install -r /tmp/requirements.txt

RUN mkdir /.cache

RUN chmod 777 /.cache
