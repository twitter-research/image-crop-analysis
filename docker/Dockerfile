FROM continuumio/miniconda3

MAINTAINER Shubhanshu Mishra <smishra@twitter.com>

ARG conda_env=image-crop-analysis

COPY ../ /workspace
WORKDIR /workspace

RUN apt-get update --fix-missing && apt-get install -y zip vim gcc g++ make
RUN conda update conda --yes
RUN conda install pip --yes
RUN pip install pandas scikit-learn scikit-image statsmodels requests dash jupyterlab
CMD ["jupyter", "lab", "--port=8900", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
