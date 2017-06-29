#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Shiva Manne <manneshiva@gmail.com>

import subprocess
import memory_profiler
import time
import sys
import os
from subprocess import call


class Train(object):
    """
    Class to train various ML frameworks for Machine/Deep Learning.

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


    def train_gensim(self):
        start_time = time.time()
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
        print cmd_str
        proc = subprocess.Popen(cmd_str.split(),
            stderr=subprocess.STDOUT,
            cwd="./nn_frameworks/gensim")  # don't escape spaces in path
        peak_mem = memory_profiler.memory_usage(proc=proc, multiprocess=True, max_usage=True)
        # delete first line from .vec file, consists of vocab size, embedding size
        # call(['sed', '-i', '1d', self.outputpath + 'gensim.vec'])
        with open(self.reportfile, 'a+') as f:
            f.write('gensim ' + str(time.time() - start_time) + ' ' + str(peak_mem) + '\n')


    def train_originalc(self):
        # no batch size parameter
        # escape spaces in path to corpus file
        start_time = time.time()
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
        print cmd_str
        proc = subprocess.Popen(cmd_str.split(),
            stderr=subprocess.STDOUT,
            cwd="./nn_frameworks/originalc")
        peak_mem = memory_profiler.memory_usage(proc=proc, multiprocess=True, max_usage=True)
        # delete first line from .vec file, consists of vocab size, embedding size
        # call(['sed', '-i', '1d', self.outputpath + 'originalc.vec'])
        print "ORIGINALC : "
        print "total training time : " + str(time.time() - start_time)
        print "peak memory : " + str(peak_mem)
        with open(self.reportfile, 'a+') as f:
            f.write('originalc ' + str(time.time() - start_time) + ' ' + str(peak_mem) + '\n')


    def train_tensorflow(self):
        # only skipgram implementation
        start_time = time.time()
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
        print cmd_str
        proc = subprocess.Popen(cmd_str.split(),
            stderr=subprocess.STDOUT,
            cwd='./nn_frameworks/tensorflow')
        peak_mem = memory_profiler.memory_usage(proc=proc, multiprocess=True, max_usage=True)
        print "TENSORFLOW : "
        print "total training time : " + str(time.time() - start_time)
        print "peak memory : " + str(peak_mem)
        with open(self.reportfile, 'a+') as f:
            f.write('tensorflow ' + str(time.time() - start_time) + ' ' + str(peak_mem) + '\n')

    def train_dl4j(self):
        cmd_str = "java -jar dl4j-examples-0.8-SNAPSHOT-jar-with-dependencies.jar"
        print cmd_str
        proc = subprocess.Popen(cmd_str.split(),
            stdout=subprocess.PIPE,
            cwd='./nn_frameworks/dl4j/dl4j-examples/dl4j-examples/target')
        peak_mem = memory_profiler.memory_usage(proc=proc, multiprocess=True, max_usage=True)
        print "DL4J : "
        print "total training time : " + str(time.time() - start_time)
        print "peak memory : " + str(peak_mem)
        # while proc.poll() is None:
        #     output = proc.stdout.readline()
        #     print output
