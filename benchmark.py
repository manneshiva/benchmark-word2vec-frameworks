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


class Train(object):
    """
    Class to train various word2vec on various ML frameworks.

    """
    def __init__(self, fname, epochs, window, size, batch_size, min_count,
                 alpha, negative, sg, workers, sample, outputpath):
        if os.path.isfile(fname):
            self.fname = fname
        else:
            raise RuntimeError('Input file does not exist at the path provided.')
        self.outputpath = outputpath
        self.epochs = epochs
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.size = size
        self.negative = negative
        self.batch_size = batch_size
        self.workers = workers
        self.sample = sample

    def train_framework(self, framework, gpu):
        """
        Method to train vectors and save metrics/report one framework at a time.

        """
        # construct command and change current working directory according to framework.

        train_dict = dict()
        cmd_str, cwd = '', ''
        if framework == 'gensim':
            cwd = '{}{}'.format(os.getcwd(), '/nn_frameworks/gensim')
            cmd_str = \
                'python gensim_word2vec.py --file {fname} --size {size} --outputpath {outputpath}' \
                ' --iter {epochs} --window {window} --min_count {min_count} --alpha {alpha} --negative {neg}' \
                ' --sg {sg} --workers {workers} --sample {sample}' \
                .format(
                    fname=self.fname, size=self.size, outputpath=self.outputpath,
                    epochs=self.epochs, window=self.window,
                    min_count=self.min_count, alpha=self.alpha, neg=self.negative,
                    sg=self.sg, workers=self.workers, sample=self.sample
                )

        elif framework == 'originalc':
            cwd = '{}{}'.format(os.getcwd(), '/nn_frameworks/originalc')
            # run executable
            cmd_str = \
                './word2vec -train {fname} -size {size} -output {outputpath} -iter {epochs}' \
                ' -window {window} -min-count {min_count} -alpha {alpha} -negative {neg} -cbow {sg} -threads' \
                ' {workers} -sample {sample} -binary 0' \
                .format(
                    fname=self.fname, size=self.size, outputpath=self.outputpath + 'originalc.vec',
                    epochs=self.epochs, window=self.window, min_count=self.min_count,
                    alpha=self.alpha, neg=self.negative, sg=not self.sg,
                    workers=self.workers, sample=self.sample
                )

        elif framework == 'tensorflow':
            cwd = '{}{}'.format(os.getcwd(), '/nn_frameworks/tensorflow')
            if gpu:
                cmd_str = \
                    'python word2vec.py --train_data {fname}' \
                    ' --embedding_size {size} --save_path_wordvectors {outputpath}' \
                    ' --epochs_to_train {epochs} --window_size {window} --min_count {min_count}' \
                    ' --learning_rate {alpha} --num_neg_samples {neg} --concurrent_steps {workers}' \
                    ' --subsample {sample} --batch_size {batch_size} --statistics_interval 5 --gpu 1' \
                    .format(
                        fname=self.fname, size=self.size, outputpath=self.outputpath + 'tensorflow-gpu.vec',
                        epochs=self.epochs, window=self.window,
                        min_count=self.min_count, alpha=self.alpha,
                        neg=self.negative, workers=self.workers,
                        sample=self.sample, batch_size=self.batch_size
                    )
            else:
                cmd_str = \
                    'python word2vec.py --train_data {fname}' \
                    ' --embedding_size {size} --save_path_wordvectors {outputpath}' \
                    ' --epochs_to_train {epochs} --window_size {window} --min_count {min_count}' \
                    ' --learning_rate {alpha} --num_neg_samples {neg} --concurrent_steps {workers}' \
                    ' --subsample {sample} --batch_size {batch_size} --statistics_interval 5' \
                    .format(
                        fname=self.fname, size=self.size, outputpath=self.outputpath + 'tensorflow.vec',
                        epochs=self.epochs, window=self.window,
                        min_count=self.min_count, alpha=self.alpha,
                        neg=self.negative, workers=self.workers,
                        sample=self.sample, batch_size=self.batch_size
                    )

        elif framework == 'dl4j':
            # run jar
            cwd = './nn_frameworks/dl4j/target'
            cmd_str = \
                'java -jar dl4j-word2vec-1.0-SNAPSHOT-jar-with-dependencies.jar' \
                ' --input {fname} --embedding_size {size} --output {outputpath} --epochs {epochs}' \
                ' --window_size {window} --min_count {min_count} --learning_rate {alpha} --neg {neg}' \
                ' --workers {workers} --subsample {sample} --batch_size {batch_size}' \
                .format(
                    fname=self.fname, size=self.size, outputpath=self.outputpath + 'dl4j.vec',
                    epochs=self.epochs, window=self.window,
                    min_count=self.min_count, alpha=self.alpha,
                    neg=self.negative, workers=self.workers, sample=self.sample,
                    batch_size=self.batch_size
                )

        logger.info('running command : %s' % cmd_str)
        # start timer
        start_time = time.time()
        proc = Popen(cmd_str.split(), stderr=STDOUT, cwd=cwd)
        peak_mem = memory_profiler.memory_usage(proc=proc, multiprocess=True, max_usage=True)
        end_time = time.time()
        #  save time and peak memory

        if gpu:
            framework = '{}-gpu'.format(framework)

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


def get_gpu_info():
    """
    Get gpu information.
    """
    gpuinfo = check_output('nvidia-smi -q', shell=True).strip()
    gpuinfo = gpuinfo.replace(':', '\n').split('\n')
    gpuinfo = [x.strip() for x in gpuinfo]
    gpuinfo_str = 'GPU INFO\n'
    gpuinfo_str += 'Model Name : {}, '.format(gpuinfo[gpuinfo.index('Product Name') + 1])
    gpuinfo_str += 'Total FB Memory : {}, '.format(gpuinfo[gpuinfo.index('FB Memory Usage') + 2])
    cuda_version = check_output('cat /usr/local/cuda/version.txt', shell=True).strip()
    gpuinfo_str += 'CUDA Version : {}'.format(cuda_version)
    return gpuinfo_str


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
    #  Evaluate word vectors on question-answer (analogies) task
    acc = model.accuracy(path_questions)
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
            rho = model.evaluate_word_pairs(os.path.join(path_word_pairs, filename))[1][0]
        except:
            rho = model.evaluate_word_pairs(os.path.join(path_word_pairs, filename), delimiter=' ')[1][0]
        eval_dict['wordpairs'][framework].append((filename, rho))

    return eval_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', help='Path to text corpus', required=True)
    parser.add_argument(
        '--frameworks', nargs='*', default=['tensorflow', 'originalc', 'dl4j', 'gensim'],
        choices=['tensorflow', 'originalc', 'dl4j', 'gensim'],
        help='Specify frameworks to run the benchmarks on(demilited by space). '
        'If None provided, benchmarks will be run on all supported frameworks.'
    )
    parser.add_argument(
        '--epochs', required=True, type=int,
        help='Number of iterations (epochs) over the corpus.'
    )
    parser.add_argument(
        '--size', required=True, type=int,
        help='Dimensionality of the embeddings/feature vectors.'
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
        '--workers', required=True, type=int,
        help='Use these many worker threads to train the model.'
    )
    parser.add_argument(
        '--sample', required=True, type=float,
        help='Set threshold for occurrence of words. Those that appear with higher frequency '
        'in the training data will be randomly down-sampled; useful range is (0, 1e-5)'
    )
    parser.add_argument(
        '--sg', required=True, choices=[0, 1], type=int,
        help='Use the skip-gram model; (use 0 for continuous bag of words model)'
    )
    parser.add_argument(
        '--negative', required=True, type=int,
        help='Number of negative examples; common values are 3 - 10 (0 = not used)'
    )
    parser.add_argument(
        '--batch_size', required=True, type=int,
        help='Mini batch size for training.'
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
    params.pop('frameworks')
    params['outputpath'] = '{}/{}'.format(os.getcwd(), TRAINED_VEC_SAVE_DIR)
    params.pop('platform')
    return params


def check_gpu():
    try:
        check_output('nvidia-smi', shell=True)
        return 1
    except:
        return 0


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

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # add console handler at INFO level
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # add file handler at DEBUG level
    handler = logging.FileHandler(os.path.join(os.getcwd(), 'benchmark.log'), 'w')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # disable overwhelming logs from eval_word_vectors function
    eval_logger = logging.getLogger('{0}.{1}'.format(__name__, 'eval_word_vectors'))
    eval_logger.setLevel(logging.ERROR)

    options = parse_args()
    report_dict = dict()

    TRAINED_VEC_SAVE_DIR = 'persistent/results/'
    QA_FILE_PATH = 'data/questions-words.txt'
    WORD_PAIRS_DIR = 'data/word-sim/'
    REPORT_FILE = "{}{}-report.json".format(TRAINED_VEC_SAVE_DIR, options.platform)

    GPU = check_gpu()  # indicates if gpu capability exists
    FRAMEWORKS_GPU = ['tensorflow']

    # get params required for training
    params = prepare_params(options)
    train = Train(**params)
    clear_trained_vecs(REPORT_FILE, TRAINED_VEC_SAVE_DIR)

    # store system information
    report_dict['systeminfo'] = get_cpu_info()

    # store gpu information, if gpu capability exists
    if GPU:
        report_dict['systeminfo'] += '\n{}'.format(get_gpu_info())

    # write config_str/model parameters to a file - useful for showing training params in the final plots
    report_dict['trainingparams'] = vars(options)
    if GPU and 'tensorflow' in options.frameworks:
        report_dict['frameworks'] = options.frameworks + ['tensorflow-gpu']
    else:
        report_dict['frameworks'] = options.frameworks
    report_dict['platform'] = options.platform

    # train and evaluate one framework at a time
    for framework in options.frameworks:
        train_dict = train.train_framework(framework, 0)
        report_dict = update_dict(report_dict, train_dict)
        logger.info('Evaluating trained word vectors\' quality for %s...' % framework)
        eval_dict = eval_word_vectors(QA_FILE_PATH, WORD_PAIRS_DIR, framework, TRAINED_VEC_SAVE_DIR)
        report_dict = update_dict(report_dict, eval_dict)
        # write report as a json string to a file
        # save after every framework to keep results in case code breaks
        with open(REPORT_FILE, 'w') as f:
            f.write(json.dumps(report_dict, indent=4))
        # train gpu implementation if gpu exists
        if GPU and framework in FRAMEWORKS_GPU:
            train_dict = train.train_framework(framework, 1)
            report_dict = update_dict(report_dict, train_dict)
            logger.info('Evaluating trained word vectors\' quality for %s-gpu...' % framework)
            eval_dict = eval_word_vectors(
                QA_FILE_PATH, WORD_PAIRS_DIR, '{}-gpu'.format(framework), TRAINED_VEC_SAVE_DIR
            )
            report_dict = update_dict(report_dict, eval_dict)
            with open(REPORT_FILE, 'w') as f:
                f.write(json.dumps(report_dict, indent=4))

    logger.info('Trained all frameworks!')
    logger.info('Reports generated!')
    logger.info('Finished running the benchmark!!!')
