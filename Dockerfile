FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

LABEL image.author.name "Josh Lamstein"
LABEL image.author.email "josh.lamstein@gladstone.ucsf.edu"


RUN conda install pandas 
RUN conda install opencv 
RUN conda install imageio 
RUN conda install sqlalchemy
RUN conda install psycopg2

ENV GUROBI_HOME="/opt/gurobi1002/linux64"
ENV PATH="${PATH}:${GUROBI_HOME}/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
ENV PATH /opt/conda/envs/datastudy/bin:$PATH
