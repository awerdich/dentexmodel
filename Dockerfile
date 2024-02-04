FROM nvidia/cuda:12.1.1-base-ubuntu22.04 AS base

ARG DEV_dentexmodel

ENV \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_SRC=/src \
    PIPENV_HIDE_EMOJIS=true \
    NO_COLOR=true \
    PIPENV_NOSPIN=true

EXPOSE 8888
RUN mkdir -p /app
WORKDIR /app

# System dependencies
RUN apt-get update -y && \
    apt-get install -y \
    'python3.10' \
    'python3-pip' \
    'git' \
    'libgl1-mesa-glx' \
    'ffmpeg' \
    'libsm6' \
    'libxext6'


# Pip and pipenv
RUN pip install --upgrade pip
RUN pip install pipenv

# Some package stuff
COPY setup.py ./
COPY src/dentexmodel/__init__.py src/dentexmodel/__init__.py

# Install dependencies into system python
COPY Pipfile Pipfile.lock ./
RUN pipenv install --system --deploy --ignore-pipfile --dev

# Run the jupyter lab server
CMD ["/bin/bash", "/app/bash_scripts/docker_entry.sh"]