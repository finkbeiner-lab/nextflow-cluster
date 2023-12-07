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
RUN apt-get install libpq5 -y
RUN Rscript -e 'install.packages("RPostgres", repos="https://cloud.r-project.org")'
RUN Rscript -e 'install.packages("tidyverse", repos="https://cloud.r-project.org")'
RUN Rscript -e 'install.packages("jsonlite", repos="https://cloud.r-project.org")'

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

RUN conda install matplotlib
RUN conda install openpyxl
RUN pip install pyomo
RUN conda install -c gurobi gurobi
# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

RUN wget -q https://downloads.imagej.net/fiji/latest/fiji-nojre.zip  && unzip fiji-nojre.zip  && rm fiji-nojre.zip

RUN wget https://dlcdn.apache.org/maven/maven-3/3.9.5/binaries/apache-maven-3.9.5-bin.tar.gz
RUN tar -xvf apache-maven-3.9.5-bin.tar.gz
RUN mv apache-maven-3.9.5 /opt/
ENV M2_HOME='/opt/apache-maven-3.9.5'
ENV PATH="$M2_HOME/bin:$PATH"
RUN export PATH
RUN wget 'https://gladstone.org/sites/default/files/styles/investigator_list/public/user-pics/investigators/finkbeiner-profile.jpg'
RUN wget "https://wsr.imagej.net/jars/ij.jar"
RUN mvn -version
RUN mvn install:install-file -Dfile=/workspace/ij.jar -DgroupId=com.finkbeiner -DartifactId=ij -Dversion=1.0 -Dpackaging=jar

ENV GUROBI_HOME="/opt/gurobi"
ENV PATH="${PATH}:${GUROBI_HOME}/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
ENV PATH /opt/conda/envs/datastudy/bin:$PATH
ENV NUMBA_CACHE_DIR=/tmp
RUN mkdir /.cellpose
RUN chmod 777 /.cellpose