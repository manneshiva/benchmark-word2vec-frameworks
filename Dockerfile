# Use the official tensorflow-cpu as a base image
# Pinned image version using it's digest
FROM tensorflow/tensorflow@sha256:32333ab56f881c52482c522934f0a0c0f9c0b7cfd46dde20e9655623fba1b9d4

WORKDIR "/"

#ADD . /

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
        gensim==2.1.0 \
        memory_profiler==0.47 \
        psutil==5.2.2

# Run bash when the container launches
CMD /bin/bash

