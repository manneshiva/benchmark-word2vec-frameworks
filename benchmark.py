#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Shiva Manne <manneshiva@gmail.com>

from train import Train

from hyperparams import Hyperparams as hp

if __name__ == '__main__':
    train = Train(epochs=10)
    train.train_gensim()
