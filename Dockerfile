# Use an official tensorflow-cpu as a base image
FROM tensorflow/tensorflow

WORKDIR "/"

# ADD . /

#setting up Java & Maven for dl4j
RUN add-apt-repository -y ppa:webupd8team/java && \ 
    apt-get update && \ 
    echo debconf shared/accepted-oracle-license-v1-1 select true | debconf-set-selections && \ 
    echo debconf shared/accepted-oracle-license-v1-1 seen true | debconf-set-selections && \ 
    apt-get install -y oracle-java8-installer && \ 
    apt-get install -y libopenjfx-java && \ 
    rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME /usr/lib/jvm/java-8-oracle
ENV JAVAFX_HOME /usr/share/java/openjfx/jre/lib/ext/

RUN wget http://redrockdigimark.com/apachemirror/maven/maven-3/3.5.0/binaries/apache-maven-3.5.0-bin.tar.gz -P /maven/ && \ 
    tar -zxf /maven/apache-maven-3.5.0-bin.tar.gz -C /maven && \ 
    rm -rf /maven/apache-maven-3.5.0-bin.tar.gz 
ENV PATH /maven/apache-maven-3.5.0/bin:$PATH

# Install python packages
RUN pip --no-cache-dir install \
        gensim \
        memory_profiler \
        psutil

# Run bash when the container launches
CMD /bin/bash

