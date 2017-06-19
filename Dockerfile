	# Use an official Python runtime as a base image
FROM tensorflow/tensorflow

WORKDIR /

ADD . /

# Install any needed packages specified in requirements.txt
RUN pip install -r /requirements.txt

# Run gensim_w2v_benchmark.py when the container launches
# CMD ["python", "/gensim/gensim_w2v_benchmark.py"]

#setting up Java - https://hub.docker.com/r/picoded/ubuntu-openjdk-8-jdk/~/dockerfile/
RUN apt-get update && \ 
	apt-get install -y openjdk-8-jre && \ 
	rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64
