FROM tensorflow/tensorflow:2.0.1-gpu-py3

ADD requirements.txt /tmp

RUN apt-get update

RUN apt-get install -y libsm6 libxext6 libxrender1

RUN pip3 install -r /tmp/requirements.txt

ADD docker/jupyter_notebook_config.json /root/.jupyter/
