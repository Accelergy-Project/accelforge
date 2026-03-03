FROM timeloopaccelergy/accelforge:latest AS base

ENV PIP_NO_CACHE_DIR=1

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install \
        seaborn \
        jupyterlab \
        ipywidgets \
        scikit-learn \
        tensorboard \
        einops && \
    pip install \
        torch \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cu124
