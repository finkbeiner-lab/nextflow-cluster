FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

LABEL image.author.name "Josh Lamstein"
LABEL image.author.email "josh.lamstein@gladstone.ucsf.edu"

# opencv dependency
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install git -y

RUN conda install pandas imageio psycopg2 scikit-image
RUN conda install -c conda-forge 'sqlalchemy>=2.0.4'
RUN conda install -c conda-forge opencv

ENV GUROBI_HOME="/opt/gurobi1002/linux64"
ENV PATH="${PATH}:${GUROBI_HOME}/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
ENV PATH /opt/conda/envs/datastudy/bin:$PATH
