# benchmark_ml_frameworks [WIP]
A repository for benchmarking popular Neural Networks frameworks used to learn unsupervised word embeddings, on different hardware platforms.
This repository is a result of [this](https://github.com/RaRe-Technologies/gensim/issues/1418) issue raised on the [Gensim repo](https://github.com/RaRe-Technologies/gensim) which achieves the following objectives:
- Compare [word2vec](https://arxiv.org/pdf/1301.3781.pdf) implementations of popular neural network frameworks by reporting metrics like time to train, peak memory, word vectors quality etc.
- Benchmark these implementaions across various cloud platforms
- Carry out the benchmarking in self-contained fully reproducible scripts for repeatable deployment
