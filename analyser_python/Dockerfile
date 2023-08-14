# For more information, please refer to https://aka.ms/vscode-docker-python
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update --fix-missing -y && apt install python3-pip npm git libsndfile1-dev  python3-numpy ffmpeg wget -y

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

ENV NUMBA_CACHE_DIR=/tmp/

# Install pip requirements

RUN python3 -m pip install imageio
RUN python3 -m pip install "imageio[pyav]"
RUN python3 -m pip install "imageio[ffmpeg]"
RUN python3 -m pip install msgpack
RUN python3 -m pip install numpy
RUN python3 -m pip install PyYAML
RUN python3 -m pip install redis 
RUN python3 -m pip install requests 
RUN python3 -m pip install grpcio==1.47.0
RUN python3 -m pip install grpcio-tools==1.47.0

RUN python3 -m pip install msgpack-numpy

RUN python3 -m pip install av

RUN python3 -m pip install pytest


RUN mkdir -p /data
WORKDIR /app/analyser/analyser
COPY ./src/ /app/analyser/analyser/

ENV PYTHONPATH="/app/"

# Switching to a non-root user, please refer to https://aka.ms/vscode-docker-python-user-rights
RUN useradd appuser 
RUN chown -R appuser /app/analyser/analyser/
RUN chown -R appuser /data
USER appuser
COPY ./config.yml /app/analyser/

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python3", "/app/analyser/analyser/server.py", "--config", "/app/analyser/config.yml", "-v"]
# CMD ["python", "backend.py"]
