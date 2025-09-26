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

# Install essential dependencies
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip graphviz make git

# Create user and group
RUN echo "**** create container user and make folders ****" && \
    userdel user || true && \
    useradd -u 911 -U -d /home/user -s /bin/bash user && \
    usermod -G users user

# Build and install islpy
WORKDIR /usr/local/src
RUN git clone https://github.com/inducer/islpy.git /usr/local/src/islpy && \
    cd /usr/local/src/islpy && \
    git submodule update --init --recursive
WORKDIR /usr/local/src/islpy/isl
RUN apt-get update && apt-get install -y --no-install-recommends autoconf automake libtool pkg-config libgmp-dev
RUN ./autogen.sh && \
    ./configure --prefix=/usr/local && \
    make -j$(nproc) && \
    make install
RUN apt-get update && apt-get install -y --no-install-recommends cmake gcc g++ python3-dev
RUN pip3 install scikit-build-core nanobind typing_extensions pcpp --break-system-packages
RUN pip3 install /usr/local/src/islpy --no-build-isolation --break-system-packages
    
# # Install essential Python build dependencies
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1
# RUN ln -s /usr/local/bin/pip3.12 /usr/local/bin/pip3
# RUN ln -s /usr/local/bin/python3.12 /usr/local/bin/python3

# RUN git clone https://github.com/gilbertmike/combinatorics.git
# RUN pip3 install ./combinatorics
# RUN git clone https://github.com/Accelergy-Project/fastfusion.git
# RUN pip3 install ./fastfusion
# RUN git clone --recurse-submodules https://github.com/Accelergy-Project/hwcomponents.git

# Install jupyterlab and ipywidgets
RUN pip3 install --break-system-packages jupyterlab ipywidgets

USER user
WORKDIR /home/user/

# Copy first_install.sh from host
COPY .dependencies/first_install.sh /.first_install.sh

# Add first_install.sh to user's .bashrc. 
# RUN echo "source /.first_install.sh" >> ~/.bashrc

# Set up entrypoint

EXPOSE 8888
# , "-c", "source /home/workspace/first_install.sh && /bin/bash"]
CMD bash -c "bash /.first_install.sh && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.notebook_dir=/home/workspace"