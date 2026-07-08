FROM docker.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    rsync \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/AtypEmu

COPY docker/requirements.txt /tmp/requirements.txt
COPY pyproject.toml ./pyproject.toml
COPY plan.md ./plan.md
COPY src ./src
COPY tests ./tests
COPY theory ./theory

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /tmp/requirements.txt && \
    python -m pip install -e .

ENV PYTHONPATH="/workspace/AtypEmu/src:${PYTHONPATH}"

CMD ["python", "-c", "import atypemu, bioemu, streamlit; print('AtypEmu BioEmu container ready')"]
