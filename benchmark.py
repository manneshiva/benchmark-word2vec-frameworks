#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Shiva Manne <manneshiva@gmail.com>

from train import Train
import argparse
import sys
import os
from textwrap import wrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




TIME_MEM_REPORT_FILENAME = 'report/time&peakmemoryReport.txt'
TIME_MEM_FIG_FILENAME = 'report/time&peakmemoryFig.jpg'
TRAINED_VEC_SAVE_DIR = 'trainedVectors/'

# Clear old reports
if os.path.isfile(TIME_MEM_REPORT_FILENAME):
    os.remove(TIME_MEM_REPORT_FILENAME)
if os.path.isfile(TIME_MEM_FIG_FILENAME):
    os.remove(TIME_MEM_FIG_FILENAME)

def plot_time_peak_mem(fname, config_str, fig_fname):
    """
    Plot the training time and peak memory reports and save to a .jpg figure

    """
    with open(fname, 'r') as f:
        lines = [line.rstrip('\n').split() for line in f]
    results = zip(*lines[:-1])  # leave last line, contain just a new line
    frameworks = results[0]
    train_times = results[1]
    peak_mems = results[2]

    fig = plt.figure(figsize=(10, 15))
    ax = fig.add_subplot(111)
    width = 0.25
    pos = [(i, i + width, i + 2 * width) for i in range(2)]
    colors = ['red', 'black', 'yellow', 'blue', 'green']
    acc_ax = ax.twinx()
    # Training time
    ax.bar(pos[0],
            train_times,
            width,
            alpha=0.5,
            color=colors
            )
    # Peak Memory
    acc_ax.bar(pos[1],
            peak_mems,
            width,
            alpha=0.5,
            color=colors
            )
    ax.set_title('Time-Memory Report')
    ax.set_ylabel('Training time (seconds)')
    acc_ax.set_ylabel('Peak Memory (MB)')
    ax.set_xlabel("\n".join(wrap(config_str, 60)))  # wrap xlabel text
    acc_ax.set_xticks([p[0] + 1.5 * width for p in pos])
    acc_ax.set_xticklabels([''] * 2)  # use empty labels to hide xticklabels
    # Proxy plots for adding legend correctly
    proxies = [ax.bar([0], [0], width=0, color=colors[i], alpha=0.5)[0] for i in range(len(frameworks))]
    plt.legend((proxies), frameworks, loc='best')
    plt.grid()
    plt.savefig(fig_fname)
    return

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

if __name__ == '__main__':
    print sys.argv
    options = parse_args(sys.argv[1:])
    print options
    print options.frameworks
    # convert options to a string of key,value pairs(config_str)
    params = vars(options).copy()  # ensure original options not modified
    params.pop('frameworks')
    params.pop('log_level')
    params['outputpath'] = TRAINED_VEC_SAVE_DIR
    params['reportfile'] = TIME_MEM_REPORT_FILENAME
    print params
    train = Train(**params)
    # print options.frameworks
    
    for framework in options.frameworks:
        getattr(train, 'train_' + framework)()
    # config_str = ', '.join("%s=%r" % (key, val) for (key, val) in options_dict.iteritems())
    # plot_time_peak_mem(TIME_MEM_REPORT_FILENAME, config_str, TIME_MEM_FIG_FILENAME)
    # print options
