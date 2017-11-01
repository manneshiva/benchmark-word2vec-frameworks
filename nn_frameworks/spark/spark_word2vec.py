#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Shiva Manne <manneshiva@gmail.com>
# Usage : `spark-submit word2vec.py --file hdfs:///data/text8-split-1000-100 --outputfile spark-test.vec --iter 1 --size 100 --window 5 --min_count 5 --partitions 8 --sample 0.01 --alpha 0.025`

import logging
import argparse
import os
from pyspark import SparkContext
from pyspark.mllib.feature import Word2Vec
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# add console handler at INFO level
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
# add file handler at DEBUG level
handler = logging.FileHandler(os.path.join(os.getcwd(), 'spark.log'), 'w')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='Path to text corpus. eg. hdfs:///data/text8', required=True)
    parser.add_argument('--outputfile', help='File to save trained vectors', required=True)
    parser.add_argument(
        '--iter', default=5, type=int,
        help='Number of iterations (epochs) over the corpus. Default : %(default)s'
    )
    parser.add_argument(
        '--size', default=100, type=int,
        help='Dimensionality of the embeddings/feature vectors. Default : %(default)s'
    )
    parser.add_argument(
        '--window', default=5, type=int,
        help='Maximum distance between the current and predicted word within a sentence. Default : %(default)s'
    )
    parser.add_argument(
        '--min_count', default=5, type=int,
        help='This will discard words that appear less than MIN_COUNT times. Default : %(default)s'
    )
    parser.add_argument(
        '--partitions', default=12, type=int,
        help='Sets number of partitions. Default : %(default)s'
    )
    parser.add_argument(
        '--alpha', default=0.025, type=float,
        help='The initial learning rate. Default : %(default)s'
    )
    return parser.parse_args()

if __name__ == '__main__':

    params = vars(parse_args())

    sc = SparkContext(appName='Word2Vec')
    inp = sc.textFile(params['file']).map(lambda row: row.split(' '))
    model = Word2Vec() \
            .setVectorSize(params['size']) \
            .setLearningRate(params['alpha']) \
            .setNumPartitions(params['partitions']) \
            .setNumIterations(params['iter']) \
            .setSeed(42) \
            .setMinCount(params['min_count']) \
            .setWindowSize(params['window']) \
            .fit(inp)
    logger.info('Saving trained model.')
    model.save(sc, 'hdfs:///model.mdl')
    logger.info('Fetching and Writing tranied vectors.')
    vecs = model.getVectors()
    with open(params['outputfile'], 'w') as f:
        word_vectors = vecs.items()
        f.write('{0} {1}\n'.format(len(word_vectors), len(word_vectors[0][1])))
        for x in vecs.items():
            f.write(x[0].encode('utf-8') + ' ')
            f.write(' '.join([str(n) for n in x[1]]))
            f.write('\n')
