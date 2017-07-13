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
DL4J | Deeplearning4j is a commercial-grade, open-source, distributed deep-learning library written for Java and Scala. Integrated with Hadoop and Spark, DL4J is designed to be used in business environments on distributed GPUs and CPUs. | Modified word2vec example given on the official github repo | [link to github example](https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/nlp/word2vec)
 
 ## Running benchmark
 
 ### Docker
 For the purpose of reproducibilty, the benchmarks need to be run inside a [docker](https://docs.docker.com/).
 - Build your own docker using the `Docker` file provided in the repo. This step assumes you have [installed docker](https://docs.docker.com/engine/installation/linux/ubuntu/#install-using-the-repository). Run the following command inside the repo directory.
 
 `sudo docker build -t benchmark_ml_frameworks:cpu .`
 
 - Download the pre-built docker image from Docker's public [registry](https://cloud.docker.com/).
 
 `sudo docker pull [TODO]`
 
 ##### GPU benchmarks [TODO: add Docker.gpu]
 Exploiting GPU to train word vectors inside docker requires `nvidia-docker`. Requirements, installation guide and why it is necessary can be found [here](https://github.com/NVIDIA/nvidia-docker). Follow the same steps mentioned above replacing `docker` with `nvidia-docker` and `benchmark_ml_frameworks:cpu` with `benchmark_ml_frameworks:gpu`.
 
 ```sudo nvidia-docker build -f Dockerfile.gpu -t benchmark_ml_frameworks:gpu .```
 
 
**Run the docker image**:
 
 `sudo docker run --rm -it benchmark_ml_frameworks:cpu` or
 
 `sudo nvidia-docker run --rm -it benchmark_ml_frameworks:gpu`

 ### Run
 To run all the benchmarks, eg:
 
 `python benchmark.py --file data/text8 --frameworks tensorflow gensim originalc --epochs 5 --size 100 --workers 4`
 
 *usage help* : `python benchmark.py -h`
 
 Available options:
 
Parameter | Description
-------- | ---
 --frameworks | Specify frameworks to run the benchmarks on(demilited by space). If None provided, benchmarks will be run on all supported frameworks.
 --file | Path to text corpus
 --epochs | Number of iterations (epochs) over the corpus. Default : 5
 --size | Dimensionality of the embeddings/feature vectors. Default : 100
 --window | Maximum distance between the current and predicted word within a sentence. Default : 5
 --min_count |  This will discard words that appear less than MIN_COUNT times. Default : 5
 --workers |  Use these many worker threads to train the model. default : 5
 --sample | Set threshold for occurrence of words. Those that appear with higher frequency in the training data will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)
--sg | Use the skip-gram model; default is 1 (use 0 for continuous bag of words model)
--negative | Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)
--alpha | The initial learning rate. Default : 0.025

On completion, the reports can be found in `report/` folder.

### Jupyter Notebook
Assuming you have cloned the repo and are currently in the directory containing this readme file, run the command:

```sudo docker run -v absPathTo/benchmark_ml_frameworks/:/benchmark_ml_frameworks/ --rm -it -p 8888:8888 benchmark_ml_frameworks:cpu ```

Once inside the docker:

```cd benchmark_ml_framework ```

Fire up the notebook, go to localhost:8888 and open run_benchmark.ipynb

```jupyter notebook --allow-root ```

  ## Running benchmark on cloud
  [TODO]
