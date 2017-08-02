#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Shiva Manne <manneshiva@gmail.com>

# from train import Train
import argparse
import sys
import os
import gensim
import memory_profiler
import time
import shutil
from textwrap import wrap
from numpy import linspace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from subprocess import call, check_output, Popen, STDOUT


REPORT_DIR = 'report/'
TRAINED_VEC_SAVE_DIR = 'trainedVectors/'
QA_FILE_PATH = 'data/questions-words.txt'
WORD_PAIRS_DIR = 'data/word-sim/'

REPORT_FILE = 'report/report.txt'
QA_REPORT = ""
WORD_PAIRS_REPORT = ""
TIME_MEM_REPORT = ""

class Train(object):
    """
    Class to train various word2vec on various ML frameworks.

    """
    def __init__(self, file, epochs, window, size, batch_size, min_count, alpha, negative, sg, workers, sample, outputpath, reportfile):
        if os.path.isfile(file):
            self.file = file
        else:
            sys.exit('Input file does not exist at the path provided.')
        self.reportfile = reportfile
        self.outputpath = outputpath
        self.epochs = epochs
        self.alpha = float(alpha)
        self.window = int(window)
        self.min_count = min_count
        self.sg = sg
        self.size = size
        self.negative = negative
        self.batch_size = batch_size
        self.workers = workers
        self.sample = sample


    def train_framework(self, framework, gpu):
        '''
        Method to train vectors and save metrics/report one framework at a time

        '''
        # construct command and change current working directory according to framework.
        if framework == 'gensim':
            cwd = './nn_frameworks/gensim'
            cmd_str = 'python gensim_word2vec.py' + ' --file ../../' + str(self.file) \
                + ' --size ' + str(self.size) \
                + ' --outputpath ../../' + str(self.outputpath) \
                + ' --iter ' + str(self.epochs) \
                + ' --window ' + str(self.window) \
                + ' --min_count ' + str(self.min_count) \
                + ' --alpha ' + str(self.alpha) \
                + ' --negative ' + str(self.negative) \
                + ' --sg ' + str(self.sg) \
                + ' --workers ' + str(self.workers) \
                + ' --sample ' + str(self.sample)

        elif framework == 'originalc':
            cwd = './nn_frameworks/originalc'
            cmd_str = './word2vec' + ' -train ../../' + str(self.file) \
                + ' -size ' + str(self.size) \
                + ' -output ../../' + str(self.outputpath) + 'originalc.vec' \
                + ' --iter ' + str(self.epochs) \
                + ' -window ' + str(self.window) \
                + ' -min-count ' + str(self.min_count) \
                + ' -alpha ' + str(self.alpha) \
                + ' -negative ' + str(self.negative) \
                + ' -cbow ' + str(int(not self.sg)) \
                + ' -threads ' + str(self.workers) \
                + ' -sample ' + str(self.sample) \
                + ' -binary 0'

        elif framework == 'tensorflow':
            cwd = './nn_frameworks/tensorflow'
            if gpu:
                cmd_str = 'python word2vec.py' + ' --train_data ../../' + str(self.file) \
                    + ' --embedding_size ' + str(self.size) \
                    + ' --save_path_wordvectors ../../' + str(self.outputpath) + 'tensorflow-gpu.vec' \
                    + ' --epochs_to_train ' + str(self.epochs) \
                    + ' --window_size ' + str(self.window) \
                    + ' --min_count ' + str(self.min_count) \
                    + ' --learning_rate ' + str(self.alpha) \
                    + ' --num_neg_samples' + str(self.negative) \
                    + ' --concurrent_steps ' + str(self.workers) \
                    + ' --subsample ' + str(self.sample) \
                    + ' --batch_size ' + str(self.batch_size) \
                    + ' --statistics_interval 5' \
                    + ' --gpu 1'
            else:
                cmd_str = 'python word2vec.py' + ' --train_data ../../' + str(self.file) \
                    + ' --embedding_size ' + str(self.size) \
                    + ' --save_path_wordvectors ../../' + str(self.outputpath) + 'tensorflow.vec' \
                    + ' --epochs_to_train ' + str(self.epochs) \
                    + ' --window_size ' + str(self.window) \
                    + ' --min_count ' + str(self.min_count) \
                    + ' --learning_rate ' + str(self.alpha) \
                    + ' --num_neg_samples' + str(self.negative) \
                    + ' --concurrent_steps ' + str(self.workers) \
                    + ' --subsample ' + str(self.sample) \
                    + ' --batch_size ' + str(self.batch_size) \
                    + ' --statistics_interval 5'
        # TODO : 1. issues while running single line text files
        elif framework == 'dl4j':
            cwd = './nn_frameworks/dl4j/target'
            cmd_str = 'java -jar dl4j-word2vec-1.0-SNAPSHOT-jar-with-dependencies.jar' + ' --input ../../../' + str(self.file) + '-split'\
                + ' --embedding_size ' + str(self.size) \
                + ' --output ../../../' + str(self.outputpath) + 'dl4j.vec' \
                + ' --epochs ' + str(self.epochs) \
                + ' --window_size ' + str(self.window) \
                + ' --min_count ' + str(self.min_count) \
                + ' --learning_rate ' + str(self.alpha) \
                + ' --neg ' + str(self.negative) \
                + ' --workers ' + str(self.workers) \
                + ' --subsample ' + str(self.sample) \
                + ' --batch_size ' + str(self.batch_size)
        print cmd_str

        # start timer
        start_time = time.time()
        proc = Popen(cmd_str.split(), stderr=STDOUT, cwd=cwd)
        peak_mem = memory_profiler.memory_usage(proc=proc, multiprocess=True, max_usage=True)
        #  save time and peak memory to a str
        global TIME_MEM_REPORT
        if gpu:
            TIME_MEM_REPORT += (framework + '-gpu ' + str(time.time() - start_time) + ' ' + str(peak_mem) + '\n')
        else:
            TIME_MEM_REPORT += (framework + ' ' + str(time.time() - start_time) + ' ' + str(peak_mem) + '\n')
        return


def prepare_dir_files():
    """
    Ensure directories exist and clear old report/trained vectors.
    """
    # Clear old contents of directory (if required) and create new
    if os.path.exists(REPORT_DIR):
        shutil.rmtree(REPORT_DIR)
    os.makedirs(REPORT_DIR)

    if os.path.exists(TRAINED_VEC_SAVE_DIR):
        shutil.rmtree(TRAINED_VEC_SAVE_DIR)
    os.makedirs(TRAINED_VEC_SAVE_DIR)


def get_cpu_info():
    """
     Get system processor information
    """
    info = check_output('lscpu', shell=True).strip()
    cpuinfo = [l.split(":") for l in info.split('\n')]
    cpuinfo = [(t[0], t[1].strip()) for t in cpuinfo]
    cpuinfo = dict(cpuinfo)
    # get system memory information
    info = check_output('cat /proc/meminfo', shell=True).strip()
    meminfo = [l.split(":") for l in info.split('\n')]
    meminfo = [(t[0], t[1].strip()) for t in meminfo]
    cpuinfo.update(dict(meminfo))

    info_keys = ['Model name', 'Architecture', 'CPU(s)', 'MemTotal']
    machine_info = 'CPU INFO\n'
    for k in info_keys:
        machine_info += (k + ":" + cpuinfo[k] + " , ")
    return machine_info


def get_gpu_info():
    """
    Get gpu information
    """
    gpuinfo = check_output('nvidia-smi -q', shell=True).strip()
    gpuinfo = gpuinfo.replace(':', '\n').split('\n')
    gpuinfo = [x.strip() for x in gpuinfo]
    gpuinfo_str = 'GPU INFO\n'
    gpuinfo_str += ('Model Name : ' + gpuinfo[gpuinfo.index('Product Name') + 1] + ', ')
    gpuinfo_str += ('Total FB Memory : ' + gpuinfo[gpuinfo.index('FB Memory Usage') + 2])
    return gpuinfo_str


def eval_word_vectors(pathQuestions, pathWordPairs, framework, trainedvectordir):
    """
    Evaluate the trained word vectors.
    """
    global QA_REPORT
    global WORD_PAIRS_REPORT
    # load trained vectors
    model = gensim.models.KeyedVectors.load_word2vec_format(trainedvectordir + framework + '.vec')
    #  Evaluate word vectors on question-answer (analogies) task
    acc = model.accuracy(pathQuestions)
    for section in acc:
        num_correct = float(len(section['correct']))
        num_incorrect = float(len(section['incorrect']))
        if(num_correct + num_incorrect) == 0:  # if none of words present in vocab
            QA_REPORT += ("%s %s %s\n" % (framework, section['section'], str(0.0)))
        else:
            QA_REPORT += ("%s %s %s\n" % (framework, section['section'], str(100.0 * (num_correct/(num_correct + num_incorrect)))))
    #  Evaluate word vectos on word-pairs task
    for filename in sorted(os.listdir(pathWordPairs)):
        try:
            rho = model.evaluate_word_pairs(os.path.join(pathWordPairs, filename))[1][0]
        except:
            rho = model.evaluate_word_pairs(os.path.join(pathWordPairs, filename), delimiter= ' ')[1][0]
        WORD_PAIRS_REPORT += ("%s %s %s\n" % (framework, filename, rho))


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
    params['reportfile'] = REPORT_FILE
    return params


def write_report_file():
    with open(REPORT_FILE, 'a+') as f:
        f.write(TIME_MEM_REPORT)
        f.write("\n")
        f.write(QA_REPORT)
        f.write("\n")
        f.write(WORD_PAIRS_REPORT)
        f.write("\n")
        return


def check_gpu():
    try:
        check_output('which nvcc', shell=True)
        return 1
    except:
        return 0

GPU = check_gpu()  # indicates if gpu capability exists
FRAMEWORKS_GPU = ['tensorflow', 'dl4j']


if __name__ == '__main__':

    options = parse_args(sys.argv[1:])
    # get params required for training
    params = prepare_params(options)
    train = Train(**params)
    prepare_dir_files()

    # write system information to a file
    with open(REPORT_FILE, 'w+') as f:
            f.write("%s\n" % get_cpu_info())
            if(not GPU):
                f.write("\n")
    # write gpu information to a file, if gpu capability exists
    if GPU:
        with open(REPORT_FILE, 'a+') as f:
                f.write("%s\n" % get_gpu_info())
                f.write("\n")
    # write config_str/model parameters to a file - useful for showing training params in the final plots
    config_str = ', '.join("%s=%r" % (key, val) for (key, val) in vars(options).iteritems())
    with open(REPORT_FILE, 'a+') as f:
            f.write("%s\n" % (config_str))
            f.write("\n")

    # train and evaluate one framework at a time
    for framework in options.frameworks:
        train.train_framework(framework, 0)
        print 'Evaluating trained word vectors\' quality for %s...' % framework
        eval_word_vectors(QA_FILE_PATH, WORD_PAIRS_DIR, framework, TRAINED_VEC_SAVE_DIR)
        # train gpu implementation if gpu exists
        if GPU and framework in FRAMEWORKS_GPU:
            train.train_framework(framework, 1)
            print 'Evaluating trained word vectors\' quality for %s-gpu...' % framework
            eval_word_vectors(QA_FILE_PATH, WORD_PAIRS_DIR, framework + '-gpu', TRAINED_VEC_SAVE_DIR)
    # write trained vectors' evaluation to report file
    write_report_file()

    print 'Trained all frameworks!'
    print 'Reports generated!'
    print 'Finished running the benchmark!!!'
