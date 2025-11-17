#
# Main image
#
FROM ubuntu:24.04

LABEL maintainer="AUTHOR EMAIL"

# Arguments
ARG BUILD_DATE
ARG VCS_REF
ARG BUILD_VERSION

# Labels
LABEL org.label-schema.schema-version="1.0"
LABEL org.label-schema.build-date=$BUILD_DATE
LABEL org.label-schema.name="FastFusion Infrastructure"
LABEL org.label-schema.description="Infrastructure FastFusion"
LABEL org.label-schema.url="https://github.com/Accelergy-Project/fastfusion/"
LABEL org.label-schema.vcs-url="https://github.com/Accelergy-Project/fastfusion"
LABEL org.label-schema.vcs-ref=$VCS_REF
LABEL org.label-schema.vendor="Author Here"
LABEL org.label-schema.version=$BUILD_VERSION
LABEL org.label-schema.docker.cmd="docker run -it --rm -v ~/workspace:/home/workspace fastfusion/fastfusion-infrastructure"

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
            # Python
            python3 \
            git \
            python3-pip \
            python3-dev \
            graphviz \
            # Build tools
            make \
            wget \
            curl \
            gcc \
            g++ \
            libgmp-dev \
            libntl-dev \
            autoconf \
            automake \
            libtool \
            pkg-config \
            make \
            && rm -rf /var/lib/apt/lists/*

# Update certificates (needed for downloading)
RUN apt-get upgrade -y ca-certificates && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/build
COPY Makefile ./
# ADD https://libntl.org/ntl-11.5.1.tar.gz /home/build/sources/ntl-11.5.1.tar.gz
# ADD https://barvinok.sourceforge.io/barvinok-0.41.8.tar.gz /home/build/sources/barvinok-0.41.8.tar.gz
# ADD https://github.com/inducer/islpy/archive/refs/tags/v2024.2.tar.gz /home/build/sources/islpy-2024.2.tar.gz

# --- build + install all ---
# RUN make install-ntl
# RUN make install-barvinok
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# RUN apt-get update && apt-get install -y python-is-python3
# RUN git clone --recurse-submodules https://github.com/inducer/islpy.git
# RUN cd islpy && LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH ./build-with-barvinok.sh /usr/local

# RUN make install-islpy
RUN make install-hwcomponents

# Install jupyterlab and ipywidgets
RUN pip install jupyterlab ipywidgets

# WORKDIR /home/workspace

# ENTRYPOINT ["/bin/bash"]

EXPOSE 8888

CMD bash -c "if ! pip list | grep -q 'fastfusion'; then cd /home/workspace && pip install -e .; fi && \
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --NotebookApp.token='' --NotebookApp.notebook_dir=/home/workspace/notebooks"

# One-liner to docker container rm -f the container that has 8888 port
# docker ps | grep 8888 | awk '{print $1}' | xargs docker rm -f