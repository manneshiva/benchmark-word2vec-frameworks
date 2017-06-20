#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Shiva Manne <manneshiva@gmail.com>

import subprocess


class Train(object):
    """
    Class to train various ML frameworks for Machine/Deep Learning.

    """
    def __init__(self, input_file="../../data/text8", epochs=5, window=5, emb=100, batch_size=32, min_count=5, alpha=0.025, negative=5, sg=1):
        self.input_file = input_file
        self.epochs = epochs
        self.alpha = float(alpha)
        self.window = int(window)
        self.min_count = min_count
        self.sg = sg
        self.emb = emb
        self.negative = negative
        self.batch_size = batch_size



    def train_gensim(self):
        proc = subprocess.Popen("python gensim_w2v_benchmark.py".split(), \
            stdout=subprocess.PIPE, \
            cwd="./nn_frameworks/gensim") # don't escape spaces in path
        while proc.poll() is None:
            output = proc.stdout.readline()
            print output

    def train_originalc(self):
        # escape spaces in path to corpus file
        cmd_str = "./word2vec -train " + self.input_file + " -output vectors.bin \
            -cbow 0 -size 100 -window 5 -negative 25 -hs 0 -sample 1e-4 \
            -threads 5 -binary 1 -iter 5"
        proc = subprocess.Popen(cmd_str.split(), \
            stdout=subprocess.PIPE, \
            stderr=subprocess.PIPE, \
            cwd="./nn_frameworks/originalc")
        print proc.stderr
        while proc.poll() is None:
            output = proc.stdout.readline()
            print output

    def train_tensorflow(self):
        cmd_str = "python word2vec.py --train_data " + self.input_file + " --save_path save/"
        print cmd_str
        proc = subprocess.Popen(cmd_str.split(),
            stdout=subprocess.PIPE,
            cwd='./nn_frameworks/tensorflow')
        while proc.poll() is None:
            output = proc.stdout.readline()
            print output
