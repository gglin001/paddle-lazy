# docker build --no-cache -t paddle-lazy:latest .
# docker build -t paddle-lazy:latest .

# FROM continuumio/miniconda3
# prefer to use miniforge3
# FROM condaforge/miniforge3
FROM graphcore/poplar:2.6.0

# install basic tools
RUN apt update && apt install vim wget -y

# install build tools
RUN apt install g++-8 -y

# install miniforge
ENV PATH="/opt/conda/bin:${PATH}"
ARG PATH="/opt/conda/bin:${PATH}"
RUN wget \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash /Miniforge3-Linux-x86_64.sh -b -p /opt/conda \
    && rm -f /Miniforge3-Linux-x86_64.sh
RUN conda init bash

# use conda-forge
RUN conda config --add channels conda-forge
RUN conda config --set channel_priority strict

# create new env
RUN conda create -n py37 python=3.7 -y

# make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "py37", "/bin/bash", "-c"]

# install tools
# will install sysroot_linux-64
# use g++ from apt, glibc version of conda is too high
# RUN conda install clangdev gxx libcxx
RUN conda install cmake ninja clang-tools clang-format -y

# install libs
# RUN conda install glog boost protobuf -y
RUN conda install pybind11 -y

# install other tools
RUN conda install bash-completion openssh git -y

# CI
RUN pip install pre-commit isort black yapf

# set env
# # TODO use ${CONDA_PREFIX}
# ENV LD_LIBRARY_PATH="/opt/conda/envs/py38/lib:${LD_LIBRARY_PATH}"
# ENV LIBRARY_PATH="/opt/conda/envs/py38/lib:${LIBRARY_PATH}"
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/py37/lib" >> ~/.bashrc
RUN echo "export LIBRARY_PATH=$LIBRARY_PATH:/opt/conda/envs/py37/lib" >> ~/.bashrc
RUN echo "export PROMPT_DIRTRIM=1" >> ~/.bashrc
RUN echo "conda activate py37" >> ~/.bashrc
