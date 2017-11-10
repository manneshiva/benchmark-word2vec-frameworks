# benchmark-word2vec-frameworks
A repository for benchmarking popular Neural Networks frameworks used to learn unsupervised word embeddings, on different hardware platforms.
This repository is a result of [this](https://github.com/RaRe-Technologies/gensim/issues/1418) issue raised on the [Gensim repo](https://github.com/RaRe-Technologies/gensim) which achieves the following objectives:
- Compare [word2vec](https://arxiv.org/pdf/1301.3781.pdf) implementations of popular neural network frameworks by reporting metrics like time to train, peak memory, word vectors quality etc.
- Benchmark these implementaions across various cloud platforms
- Carry out the benchmarking in self-contained fully reproducible scripts for repeatable deployment

## Frameworks
Below is a short description of the frameworks supported and the code used to train the word vectors.

Framework | Description | Code used | References
--- | --- | --- | ---
Tensorflow | TensorFlow is an open source software library for numerical computation using data flow graphs. Widely used for conducting  machine learning and deep neural networks research. Can run on both CPU or GPU. | The code used in benchmarking has been directly picked up and modified from the tutorials on the official github page. | [link to github code](https://github.com/tensorflow/models/tree/master/tutorials/embedding)
OriginalC | This tool has been made available by the original author([Tomas Mikolov](https://arxiv.org/find/cs/1/au:+Mikolov_T/0/1/0/all/0/1))and provides an efficient implementation of the continuous bag-of-words and skip-gram architectures for computing vector representations of words. | Use unmodified code and simply run the executable by setting relevant parameters | [link to code](https://github.com/tmikolov/word2vec) [link to toolkit](https://code.google.com/archive/p/word2vec/)
Gensim | Gensim is a Python library for topic modelling, document indexing and similarity retrieval with large corpora. Target audience is the natural language processing (NLP) and information retrieval (IR) community. | Refer to the code given on the word2vec tutorial on the official github repo | [link to reference code](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/word2vec.ipynb)
DL4J | Deeplearning4j is a commercial-grade, open-source, distributed deep-learning library written for Java and Scala. Integrated with Hadoop and Spark, DL4J is designed to be used in business environments on distributed GPUs and CPUs. | Modified word2vec example given on the official github repo | [link to github example](https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/nlp/word2vec)
Spark|Apache Spark is a fast and general cluster computing system for Big Data. It provides high-level APIs in Scala, Java, Python, and R, and an optimized engine that supports general computation graphs for data analysis|Tweaked official word2vec example to exploit multiple computing nodes | [link to example code](https://github.com/apache/spark/blob/master/examples/src/main/python/mllib/word2vec.py)
 
 ## Running benchmark
 
 ### Docker
 For the purpose of reproducibilty, the benchmarks need to be run inside a [docker](https://docs.docker.com/).
 - Build your own docker using the `Docker` file provided in the repo. This step assumes you have [installed docker](https://docs.docker.com/engine/installation/linux/ubuntu/#install-using-the-repository). Run the following command from the directory containing the this repository.
 
 `docker build -f Dockerfile-cpu-tfsource -t manneshiva/playground:benchmarkword2vec-cpu-tfsource .`
 
 - Download the pre-built docker image from Docker's public [registry](https://cloud.docker.com/).
 
 `docker pull manneshiva/playground:benchmarkword2vec-cpu-tfsource`
 
 [Images on Docker Hub](https://hub.docker.com/r/manneshiva/playground/tags/)

 ##### GPU benchmarks
 Exploiting GPU to train word vectors inside docker requires `nvidia-docker`. Requirements, installation guide and why it is necessary can be found [here](https://github.com/NVIDIA/nvidia-docker). Follow the same steps mentioned above replacing `docker` with `nvidia-docker` and `benchmarkword2vec-cpu-tfsource` with `benchmarkword2vec-gpu-tfsource`.
 
 ```docker build -f Dockerfile-gpu-tfsource -t manneshiva/playground:benchmarkword2vec-gpu-tfsource .```

**P.S.:** In order to benefit from sse4.2/avx/fma optimizations (which results in a faster training time for tensorflow), tensorflow has been built from source. In case your machine doesn't support this, use `benchmarkword2vec-{c/g}pu`.

 
**Run the docker image**:

`docker run -v absPathTo/persistent/:/benchmark-word2vec-frameworks/persistent/ -w /benchmark-word2vec-frameworks/ --rm -it -p 8888:8888 manneshiva/playground:benchmarkword2vec-cpu-tfsource /bin/bash`
 
 
 
 `nvidia-docker run -v absPathTo/persistent/:/benchmark-word2vec-frameworks/persistent/ -w /benchmark-word2vec-frameworks/ --rm -it -p 8888:8888 manneshiva/playground:benchmarkword2vec-gpu /bin/bash`

 ### Run
 To run all the benchmarks, eg:
 
 ```
 python benchmark.py --fname /benchmark-word2vec-frameworks/persistent/enwiki-20170501-2M.cor \
--frameworks tensorflow originalc dl4j gensim --epochs 4 --size 100 --window 5 --min_count 30 \
--negative 5 --workers 4 --sample 0.001 --sg 1 --batch_size 64 --alpha 0.025 --platform aws \
|& tee persistent/benchmark.log
 ```
 
 *usage help* : `python benchmark.py -h`
 
 Available options:
 
Parameter | Description
-------- | ---
 --frameworks | Specify frameworks to run the benchmarks on(demilited by space). If None provided, benchmarks will be run on all supported frameworks.
 --file | Path to text corpus
 --epochs | Number of iterations (epochs) over the corpus.
 --size | Dimensionality of the embeddings/feature vectors.
 --window | Maximum distance between the current and predicted word within a sentence.
 --min_count |  This will discard words that appear less than MIN_COUNT times.
 --workers |  Use these many worker threads to train the model.
 --sample | Set threshold for occurrence of words. Those that appear with higher frequency in the training data will be randomly down-sampled; useful range is (0, 1e-5)
--sg | Use the skip-gram model; default is 1 (use 0 for continuous bag of words model)
--negative | Number of negative examples; common values are 3 - 10 (0 = not used)
--alpha | The initial learning rate. eg. 0.025

On completion, the report, trained vectors and logs file can be found in `persistent/`.



### Jupyter Notebook
To generate graphics from the report,fire up the notebook, go to localhost:8888 and open visualize_report.ipynb

```jupyter notebook --allow-root ```

  ## Running benchmark on cloud
 Modify the `hosts` file in the `ansible/` folder and run the following command:
 
 `ansible-playbook -i hosts benchmark.yml -v`

This will install Docker, pull the Docker image and download the preprocessed wiki corpus(can be in this repository's [release](https://github.com/manneshiva/benchmark-word2vec-frameworks/releases)) without manual intervention. Once this command is complete, start the docker and run benchmark.
