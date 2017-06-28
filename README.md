# benchmark_ml_frameworks [WIP]
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
DL4J | Deeplearning4j is the first commercial-grade, open-source, distributed deep-learning library written for Java and Scala. Integrated with Hadoop and Spark, DL4J is designed to be used in business environments on distributed GPUs and CPUs. | Modified word2vec example given on the official github repo | [link to github example](https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/nlp/word2vec)
