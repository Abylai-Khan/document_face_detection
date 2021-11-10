FROM ubuntu:20.04

ENV LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    APP_ROOT=/app_root \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH="${PYTHONPATH}:/app_root/app"

RUN apt update && apt-get install -y \
    python3.8 python3-pip locales \
    curl wget git vim iputils-ping dnsutils\
&& apt clean \
&& rm -rf /var/lib/apt/lists/* \
&& ln -sf /usr/bin/python3.8 /usr/local/bin/python \
&& ln -sf /usr/bin/python3.8 /usr/local/bin/python3 \
&& python -m pip install -U pip setuptools poetry \
&& ln -sf /usr/local/bin/pip3 /usr/local/bin/pip \
&& locale-gen en_US.UTF-8

WORKDIR ${APP_ROOT}

# ========== Custom Settings ==========
# Please install the libraries if necessary.

# ========== END ==========

# Install python packages
COPY poetry.lock pyproject.toml ${APP_ROOT}/
RUN poetry config virtualenvs.create false && poetry install

# Copy files
COPY config ${APP_ROOT}/config
COPY data ${APP_ROOT}/data
COPY processes ${APP_ROOT}/processes
COPY podder_task_foundation ${APP_ROOT}/podder_task_foundation
COPY podder_task_foundation_plugins ${APP_ROOT}/podder_task_foundation_plugins
COPY manage.py ${APP_ROOT}/manage.py


EXPOSE 5000
ENTRYPOINT ["python", "manage.py", "http", "-H", "0.0.0.0", "-p", "5000"]
