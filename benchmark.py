#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Shiva Manne <manneshiva@gmail.com>

from train import Train

if __name__ == '__main__':
    train = Train(epochs=10)
    train.train_gensim()
    train.train_originalc()
    train.train_tensorflow()
    train.train_dl4j()
