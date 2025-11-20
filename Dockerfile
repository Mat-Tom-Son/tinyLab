FROM nvidia/cuda:12.6.0-base-debian12

ENV DEBIAN_FRONTEND=noninteractive
ENV VENV_PATH=/opt/venv
ENV TORCH_VERSION=2.9.0
ENV TORCHVISION_VERSION=0.24.0
ENV TORCHAUDIO_VERSION=2.9.0
ENV TORCH_INDEX=https://download.pytorch.org/whl/cu121

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      git build-essential python3 python3-venv python3-pip python3-dev \
      wget curl ca-certificates pkg-config libssl-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN python3 -m venv ${VENV_PATH} && \
    . ${VENV_PATH}/bin/activate && \
    python -m pip install --upgrade pip setuptools wheel && \
    pip install \
      "torch==${TORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      "torchaudio==${TORCHAUDIO_VERSION}" \
      --index-url "${TORCH_INDEX}" && \
    pip install \
      transformer-lens==2.16.1 \
      transformers==4.57.1 \
      datasets==4.3.0 \
      mlflow==3.5.1 \
      umap-learn==0.5.9.post2 \
      plotly==6.3.1 \
      kaleido==1.1.0 \
      matplotlib==3.10.7 \
      pandas==2.3.3 \
      numpy==2.3.3 \
      psutil==7.1.2 \
      orjson==3.11.4 \
      rich==14.2.0 \
      pyarrow==21.0.0 \
      pydantic==2.12.3 \
      tqdm==4.67.1 \
      sae-lens==6.20.1 \
      "dvc[gcs]>=3.51,<4" && \
    pip install -e .

ENV PATH="${VENV_PATH}/bin:${PATH}"
ENV PYTHONPATH=/app

CMD ["bash", "scripts/run_pilot_dry_run.sh"]
