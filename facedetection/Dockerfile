# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8-slim-buster

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN apt-get update -y && apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

# create working directory and copy files to it
WORKDIR /app
COPY . /app

# install requirements, execute makefile and create user
RUN cd /app && \
    python -m pip install -r requirements.txt && \
    make && \
    useradd appuser && chown -R appuser /app

USER appuser

# exectue program
CMD ["python", "facedetection_server.py"]
