FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/spuq-cartography
COPY requirements.txt /opt/ml/spuq-cartography/requirements.txt
RUN pip install -r requirements.txt
COPY ./ /opt/ml/spuq-cartography/

# ENV WANDB_PROJECT XXX
# ENV WANDB_ENTITY XXX
# ENV WANDB_API_KEY XXX

ENV HF_HOME=/opt/ml/input/data/huggingface_cache
ENV BASELINE_NUM_EPOCHS=5
ENV BASELINE_BATCH_SIZE=1
ENV PARTITIONS_NUM_EPOCHS=5
ENV PARTITIONS_BATCH_SIZE=1

ENTRYPOINT ["./run.sh"]
