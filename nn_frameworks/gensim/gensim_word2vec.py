#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Shiva Manne <manneshiva@gmail.com>

import gensim
import logging
import argparse
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='Path to text corpus', required=True)
    parser.add_argument('--outputpath', help='Path to save trained vectors', required=True)
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
        '--workers', default=3, type=int,
        help='Use these many worker threads to train the model. Default : %(default)s'
    )
    parser.add_argument(
        '--sample', default=1e-3, type=float,
        help='Set threshold for occurrence of words. Those that appear with higher frequency'
        'in the training data will be randomly down-sampled; default is %(default)s, useful range is (0, 1e-5)'
    )
    parser.add_argument(
        '--sg', default=1, choices=[0, 1], type=int,
        help='Use the skip-gram model; default is %(default)s (use 0 for continuous bag of words model)'
    )
    parser.add_argument(
        '--negative', default=5, type=int,
        help='Number of negative examples; default is %(default)s, common values are 3 - 10 (0 = not used)'
    )
    parser.add_argument(
        '--alpha', default=0.025, type=float,
        help='The initial learning rate. Default : %(default)s'
    )
    return parser.parse_args()


if __name__ == '__main__':

    params = vars(parse_args())

    sentences = gensim.models.word2vec.LineSentence(params['file'])
    params.pop('file')

    outputpath = params['outputpath']
    params.pop('outputpath')

    model = gensim.models.Word2Vec(sentences, **params)
    model.wv.save_word2vec_format(outputpath + 'gensim.vec')
