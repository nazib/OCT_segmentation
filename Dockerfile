FROM nvcr.io/nvidia/pytorch:22.11-py3
FROM continuumio/miniconda3

ADD environment.yaml /tmp/environment.yaml
RUN rm -rf /workspace/*
WORKDIR /workspace

RUN conda env create -f /tmp/environment.yaml
#SHELL ["conda", "run", "-n", "oct", "/bin/bash", "-c"]
RUN echo "conda activate oct" >> ~/.bashrc
ENV PATH /opt/conda/envs/oct/bin:$PATH
ENV CONDA_DEFAULT_ENV $oct
COPY . .
# Start the server when the container launches
CMD ["/workspace/entrypoint.sh"]