#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Shiva Manne <manneshiva@gmail.com>

from train import Train
import argparse
import sys
import os
import gensim
from textwrap import wrap
from numpy import linspace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from subprocess import call


REPORT_DIR = 'report/'
TIME_MEM_REPORT_FILENAME = 'report/time&peakmemoryReport.txt'
TIME_MEM_FIG_FILENAME = 'report/time&peakmemoryFig.jpg'
EVAL_WORD_VEC_REPORT_FILENAME = 'report/word-pairs-report.txt'
EVAL_WORD_VEC_FIG_FILENAME = 'report/word-pairs-report-fig.jpg'
TRAINED_VEC_SAVE_DIR = 'trainedVectors/'
EVAL_QA_REPORT_FILENAME = 'report/question-answer-report.txt'
EVAL_QA_FIG_FILENAME = 'report/question-answer-fig.jpg'
QA_FILE_PATH = 'data/questions-words.txt'
WORD_PAIRS_DIR = 'data/word-sim/'
PARAMS_FILENAME = 'report/training-parameters.txt'

def prepare_dir_files():
    """
    Ensure directories exist and clear old report files/figures.
    """
    # Ensure dir exists
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
    if not os.path.exists(TRAINED_VEC_SAVE_DIR):
        os.makedirs(TRAINED_VEC_SAVE_DIR)
    # Clear old reports
    if os.path.isfile(TIME_MEM_REPORT_FILENAME):
        os.remove(TIME_MEM_REPORT_FILENAME)
    if os.path.isfile(TIME_MEM_FIG_FILENAME):
        os.remove(TIME_MEM_FIG_FILENAME)
    if os.path.isfile(EVAL_WORD_VEC_REPORT_FILENAME):
        os.remove(EVAL_WORD_VEC_REPORT_FILENAME)
    if os.path.isfile(EVAL_WORD_VEC_FIG_FILENAME):
        os.remove(EVAL_WORD_VEC_FIG_FILENAME)
    if os.path.isfile(EVAL_QA_REPORT_FILENAME):
        os.remove(EVAL_QA_REPORT_FILENAME)
    if os.path.isfile(EVAL_QA_FIG_FILENAME):
        os.remove(EVAL_QA_FIG_FILENAME)


def clean_old_reports():
    # clean old reports
    if os.path.isfile(EVAL_WORD_VEC_REPORT_FILENAME):
        os.remove(EVAL_WORD_VEC_REPORT_FILENAME)
    if os.path.isfile(EVAL_WORD_VEC_FIG_FILENAME):
        os.remove(EVAL_WORD_VEC_FIG_FILENAME)
    if os.path.isfile(EVAL_QA_REPORT_FILENAME):
        os.remove(EVAL_QA_REPORT_FILENAME)
    if os.path.isfile(EVAL_QA_FIG_FILENAME):
        os.remove(EVAL_QA_FIG_FILENAME)


def eval_word_vectors(pathQuestions, pathWordPairs, framework, trainedvectordir, reportdir):
    # load trained vectors
    model = gensim.models.KeyedVectors.load_word2vec_format(trainedvectordir + framework + '.vec')
    #  Evaluate word vectors on question-answer (analogies) task
    acc = model.accuracy(pathQuestions)
    with open(reportdir + 'question-answer-report.txt', 'a+') as f:
        for section in acc:
            num_correct = float(len(section['correct']))
            num_incorrect = float(len(section['incorrect']))
            if(num_correct + num_incorrect) == 0:  # if none of words present in vocab
                f.write("%s %s %s\n" % (framework, section['section'], str(0.0)))
            else:
                f.write("%s %s %s\n" % (framework, section['section'], str(100.0 * (num_correct/(num_correct + num_incorrect)))))
    #  Evaluate word vectos on word-pairs
    with open(reportdir + 'word-pairs-report.txt', 'a+') as f:
        for filename in sorted(os.listdir(pathWordPairs)):
            try:
                rho = model.evaluate_word_pairs(os.path.join(pathWordPairs, filename))[1][0]
            except:
                rho = model.evaluate_word_pairs(os.path.join(pathWordPairs, filename), delimiter= ' ')[1][0]
            f.write("%s %s %s\n" % (framework, filename, rho))


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='Path to text corpus', required=True)
    parser.add_argument('--frameworks', nargs='*', default=['tensorflow', 'originalc','dl4j','gensim'], choices=['tensorflow', 'originalc', 'dl4j', 'gensim'], help='Specify frameworks to run the benchmarks on(demilited by space). If None provided, benchmarks will be run on all supported frameworks.')
    parser.add_argument('--epochs', default=5, type=int, help='Number of iterations (epochs) over the corpus. Default : 5')
    parser.add_argument('--size', default=100, type=int, help='Dimensionality of the embeddings/feature vectors. Default : 100')
    parser.add_argument('--window', default=5, type=int, help='Maximum distance between the current and predicted word within a sentence. Default : 5')
    parser.add_argument('--min_count', default=5, type=int, help='This will discard words that appear less than MIN_COUNT times. Default : 5')
    parser.add_argument('--workers', default=3, type=int, help='Use these many worker threads to train the model. Default : 3')
    parser.add_argument('--sample', default=1e-3, type=float, help='Set threshold for occurrence of words. Those that appear with higher frequency \
        in the training data will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)')
    parser.add_argument('--sg', default=1, choices=[0, 1], type=int, help='Use the skip-gram model; default is 1 (use 0 for continuous bag of words model)')
    parser.add_argument('--negative', default=5, type=int, help='Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)')
    parser.add_argument('--batch_size', default=32, type=int, help='Mini batch size for training. Default : 32')
    parser.add_argument('--alpha', default=0.025, type=float, help='The initial learning rate. Default : 0.025')
    parser.add_argument('--log-level', default='INFO', help='Specify logging level. Default : INFO')
    return parser.parse_args(args)


def prepare_params(options):
    params = vars(options).copy()  # ensure original options not modified
    params.pop('frameworks')
    params.pop('log_level')
    params['outputpath'] = TRAINED_VEC_SAVE_DIR
    params['reportfile'] = TIME_MEM_REPORT_FILENAME
    return params


if __name__ == '__main__':

    options = parse_args(sys.argv[1:])
    # get only those params required for training
    params = prepare_params(options)
    train = Train(**params)
    prepare_dir_files()
    clean_old_reports()
    for framework in options.frameworks:
        getattr(train, 'train_' + framework)()
        print 'Evaluating trained word vectors\' quality for %s...' % framework
        eval_word_vectors(QA_FILE_PATH, WORD_PAIRS_DIR, framework, TRAINED_VEC_SAVE_DIR, REPORT_DIR)
        # call(['python', 'eval_word_vectors/all_wordsim.py', TRAINED_VEC_SAVE_DIR + framework + '.vec', 'eval_word_vectors/data/word-sim/', framework, EVAL_WORD_VEC_REPORT_FILENAME])
    print 'Trained all frameworks!'
    print 'Reports generated!'
    # write config_str to a file - useful for showing training params in the final plots
    config_str = ', '.join("%s=%r" % (key, val) for (key, val) in vars(options).iteritems())
    with open(PARAMS_FILENAME, 'w+') as f:
            f.write("%s\n" % (config_str))
    print 'Finished running the benchmark!!!'
