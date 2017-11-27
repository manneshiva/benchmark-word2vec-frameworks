#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Shiva Manne <manneshiva@gmail.com>

import argparse
import os
import gensim
import memory_profiler
import time
import shutil
import json
from subprocess import check_output, Popen, STDOUT
import logging
import collections


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class Train(object):
    """
    Class to train various word2vec on various ML frameworks.

    """
    def __init__(self, fname, epochs, window, size, min_count,
                 alpha, partitions, outputpath):

        self.fname = fname
        self.outputpath = outputpath
        self.epochs = epochs
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.size = size
        self.partitions = partitions

    def train_framework(self, framework, gpu):
        """
        Method to train vectors and save metrics/report one framework at a time.

        """
        # construct command and change current working directory according to framework.

        train_dict = dict()
        cmd_str, cwd = '', ''
        cwd = '{}{}'.format(os.getcwd(), '/nn_frameworks/spark')
        cmd_str = \
            'spark-submit spark_word2vec.py --file {fname} --outputfile {outputfile} --size {size}' \
            ' --iter {epochs} --window {window} --min_count {min_count} --alpha {alpha}' \
            ' --partitions {partitions}' \
            .format(
                fname=self.fname, size=self.size, outputfile=self.outputpath + 'spark.vec',
                epochs=self.epochs, window=self.window, min_count=self.min_count,
                alpha=self.alpha, partitions=self.partitions
            )
        logger.info('running command : %s' % cmd_str)
        # start timer
        start_time = time.time()
        proc = Popen(cmd_str.split(), stderr=STDOUT, cwd=cwd)
        peak_mem = memory_profiler.memory_usage(proc=proc, multiprocess=True, max_usage=True)
        end_time = time.time()
        #  save time and peak memory

        train_dict['time'] = dict()
        train_dict['memory'] = dict()
        train_dict['command'] = dict()
        train_dict['time'][framework] = int(end_time - start_time)
        train_dict['memory'][framework] = int(peak_mem)
        train_dict['command'][framework] = cmd_str

        return train_dict


def clear_trained_vecs(report_dict, trained_vec_save_dir):
    """
    Ensure directory exist and clear old report/trained vectors.
    """
    # Clear old contents of directory (if required) and create new
    if os.path.exists(report_dict):
        os.remove(report_dict)
    if os.path.exists(trained_vec_save_dir):
        shutil.rmtree(trained_vec_save_dir)
    os.makedirs(trained_vec_save_dir)


def get_cpu_info():
    """
     Get system processor information.
    """
    info = check_output('lscpu', shell=True).strip().split('\n')
    cpuinfo = [l.split(":") for l in info]
    cpuinfo = [(t[0], t[1].strip()) for t in cpuinfo]
    cpuinfo = dict(cpuinfo)

    # get system memory information
    info = check_output('cat /proc/meminfo', shell=True).strip().split('\n')
    meminfo = [l.split(":") for l in info]
    meminfo = [(t[0], t[1].strip()) for t in meminfo]
    cpuinfo.update(dict(meminfo))

    info_keys = ['Model name', 'Architecture', 'CPU(s)', 'MemTotal']
    machine_info = 'CPU INFO\n'
    for k in info_keys:
        machine_info += '{}:{}, '.format(k, cpuinfo[k])
    return machine_info


def eval_word_vectors(path_questions, path_word_pairs, framework, trained_vector_dir):
    """
    Evaluate the trained word vectors.
    """
    eval_dict = dict()
    eval_dict['qa'] = dict()
    eval_dict['wordpairs'] = dict()
    eval_dict['qa'][framework] = []
    eval_dict['wordpairs'][framework] = []

    model = gensim.models.KeyedVectors.load_word2vec_format(trained_vector_dir + framework + '.vec')
    logger.info('Vocab Size : %s' % len(model.vocab))
    #  Evaluate word vectors on question-answer (analogies) task
    acc = model.accuracy(path_questions, restrict_vocab=len(model.vocab))
    for section in acc:
        num_correct = float(len(section['correct']))
        num_incorrect = float(len(section['incorrect']))
        if (num_correct + num_incorrect) == 0:  # if none of words present in vocab
            eval_dict['qa'][framework].append((section['section'], str(0.0)))
        else:
            eval_dict['qa'][framework].append(
                (section['section'], str(100.0 * (num_correct/(num_correct + num_incorrect))))
            )

    #  Evaluate word vectors on word-pairs task
    for filename in sorted(os.listdir(path_word_pairs)):
        try:
            rho = model.evaluate_word_pairs(os.path.join(path_word_pairs, filename), restrict_vocab=len(model.vocab))[1][0]
        except:
            rho = model.evaluate_word_pairs(os.path.join(path_word_pairs, filename), delimiter=' ', restrict_vocab=len(model.vocab))[1][0]
        eval_dict['wordpairs'][framework].append((filename, rho))

    return eval_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', help='Path to text corpus. eg: hdfs:///data/text8', required=True)
    parser.add_argument(
        '--epochs', required=True, type=int,
        help='Number of iterations (epochs) over the corpus.'
    )
    parser.add_argument(
        '--size', required=True, type=int,
        help='Dimensionality of the embeddings/feature vectors.'
    )
    parser.add_argument(
        '--workers', required=True, type=int,
        help='The number of spark workers.'
    )
    parser.add_argument(
        '--window', required=True, type=int,
        help='Maximum distance between the current and predicted word within a sentence.'
    )
    parser.add_argument(
        '--min_count', required=True, type=int,
        help='This will discard words that appear less than MIN_COUNT times.'
    )
    parser.add_argument(
        '--partitions', required=True, type=int,
        help='Sets number of partitions.'
    )
    parser.add_argument(
        '--alpha', required=True, type=float,
        help='The initial learning rate.'
    )
    parser.add_argument(
        '--platform', help='Platform the benchmark is being run on. eg. aws, azure', required=True
    )
    return parser.parse_args()


def prepare_params(options):
    params = vars(options).copy()  # ensure original options not modified
    params.pop('workers')
    params['outputpath'] = '{}/{}'.format(os.getcwd(), TRAINED_VEC_SAVE_DIR)
    params.pop('platform')
    return params


def update_dict(d, u):
    # merge nested dict 'u' to nested dict 'd'
    for k, v in u.iteritems():
        if isinstance(v, collections.Mapping):
            r = update_dict(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


if __name__ == '__main__':

    options = parse_args()
    report_dict = dict()

    TRAINED_VEC_SAVE_DIR = 'persistent/results/'
    QA_FILE_PATH = 'data/questions-words.txt'
    WORD_PAIRS_DIR = 'data/word-sim/'
    REPORT_FILE = "{}{}-report.json".format(TRAINED_VEC_SAVE_DIR, options.platform)

    # get params required for training
    params = prepare_params(options)
    print params
    train = Train(**params)
    clear_trained_vecs(REPORT_FILE, TRAINED_VEC_SAVE_DIR)

    # store system information
    report_dict['systeminfo'] = get_cpu_info()

    # write config_str/model parameters to a file - useful for showing training params in the final plots
    report_dict['trainingparams'] = vars(options)
    report_dict['platform'] = options.platform

    # train and evaluate spark implementation
    train_dict = train.train_framework('spark', 0)
    report_dict = update_dict(report_dict, train_dict)
    logger.info('Evaluating trained word vectors\' quality for %s...' % 'spark')
    eval_dict = eval_word_vectors(QA_FILE_PATH, WORD_PAIRS_DIR, 'spark', TRAINED_VEC_SAVE_DIR)
    report_dict = update_dict(report_dict, eval_dict)
    # write report as a json string to a file
    # save after every framework to keep results in case code breaks
    with open(REPORT_FILE, 'w') as f:
        f.write(json.dumps(report_dict, indent=4))

    logger.info('Trained all frameworks!')
    logger.info('Reports generated!')
    logger.info('Finished running the benchmark!!!')
