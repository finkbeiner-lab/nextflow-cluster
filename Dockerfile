FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

LABEL image.author.name "Josh Lamstein"
LABEL image.author.email "josh.lamstein@gladstone.ucsf.edu"

# opencv dependency
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install git -y
RUN apt-get install wget

RUN conda install pandas imageio psycopg2 scikit-image
RUN conda install -c conda-forge 'sqlalchemy>=2.0.4'
RUN python -m pip install opencv-python
RUN python -m pip install wandb
RUN python -m pip install cellpose

RUN apt install dirmngr gnupg apt-transport-https ca-certificates software-properties-common -y
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'
RUN apt install r-base -y
RUN apt install build-essential
RUN Rscript -e 'install.packages("optparse", repos="https://cloud.r-project.org")'
RUN Rscript -e 'install.packages("survival", repos="https://cloud.r-project.org")'
RUN Rscript -e 'install.packages("ggplot2", repos="https://cloud.r-project.org")'
RUN Rscript -e 'install.packages("ggfortify", repos="https://cloud.r-project.org")'
RUN Rscript -e 'install.packages("RPostgreSQL", repos="https://cloud.r-project.org")'
RUN Rscript -e 'install.packages("vscDebugger", repos="https://cloud.r-project.org")'


# Install OpenJDK-8
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;
    
# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

RUN wget "https://wsr.imagej.net/jars/ij.jar"

ENV GUROBI_HOME="/opt/gurobi1002/linux64"
ENV PATH="${PATH}:${GUROBI_HOME}/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
ENV PATH /opt/conda/envs/datastudy/bin:$PATH
