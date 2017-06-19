#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Shiva Manne <manneshiva@gmail.com>

import sys
import os
# sys.path.append(os.path.abspath('.') + '/nn_frameworks/gensim')


class Train(object):
    """
    Class to train various ML frameworks for MAchine/Deep Learning.

    """
    def __init__(self, input_file='data/text8', epochs=5):
        self.input_file = input_file
        self.epochs = epochs
        print "------init--------"
        print self.input_file
        print self.epochs
        print "-------------------"

    def train_gensim(self):
        # import gensim_w2v_benchmark
        execfile("./nn_frameworks/gensim/gensim_w2v_benchmark.py")
