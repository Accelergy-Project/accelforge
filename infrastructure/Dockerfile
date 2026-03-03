FROM timeloopaccelergy/accelforge:latest AS base

ENV PIP_NO_CACHE_DIR=1

RUN python3 -m pip install \
        seaborn \
        jupyterlab \
        ipywidgets \
        scikit-learn \
        tensorboard \
        einops
# RUN python3 -m pip install \
#         torch \
#         torchvision \
#         torchaudio \
#         --index-url https://download.pytorch.org/whl/cu124
