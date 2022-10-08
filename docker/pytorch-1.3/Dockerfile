FROM ic-registry.epfl.ch/cvlab/lis/lab-base:cuda10-devel

ENV CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda"
# Install basic utilities
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        python3 python3-dev python3-pip \
        cmake gcc make build-essential \
        libnuma-dev libblas-dev liblapack-dev openmpi-common\
        ca-certificates && \
    rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && ln -s /usr/bin/python3 /usr/local/bin/python \
    && ln -s /usr/bin/pip3 /usr/local/bin/pip
# core
RUN pip --no-cache-dir install --upgrade \
    setuptools wheel virtualenv pip
# main
RUN pip --no-cache-dir install \
    numpy scipy scikit-image scikit-learn \
    matplotlib pyyaml

RUN pip --no-cache-dir install \
    ipython jupyterlab
# opencv
RUN pip --no-cache-dir install \
    opencv-contrib-python-headless

# Build PyTorch from source
WORKDIR /opt
RUN git clone https://github.com/pytorch/pytorch.git \
    && cd pytorch \
    && git submodule update --init --recursive \
    && git checkout v1.3.0 \
    && python setup.py install \
    && rm -rf pytorch

# Build torch-vision from source
RUN git clone https://github.com/pytorch/vision.git \
    && cd vision \
    && python setup.py install \
    && rm -rf vision
