#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Shiva Manne <manneshiva@gmail.com>
from textwrap import wrap
from numpy import linspace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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

def plot_time_peak_mem(fname, fig_fname, config_str):
    """
    Plot the training time and peak memory reports and save to a .jpg figure.

    """
    with open(fname, 'r') as f:
        lines = [line.rstrip('\n').split() for line in f]
    results = zip(*lines[:])
    frameworks = results[0]
    train_times = results[1]
    peak_mems = results[2]
    print frameworks
    fig = plt.figure(figsize=(10, 15))
    ax = fig.add_subplot(111)
    pos = [linspace(i + 0.25, i + 0.75, num=len(frameworks), endpoint=False) for i in range(2)]
    width = pos[0][1] - pos[0][0]
    colors = ['red', 'black', 'yellow', 'blue', 'green', 'orange', 'grey']
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


def plot_word_eval_report(fname, fig_fname, title, analogies=True):
    """
    Plot the eval-word-vectors report and save to a .jpg figure.

    """
    with open(fname, 'r') as f:
        lines = [line.rstrip('\n').split() for line in f]
    results = zip(*lines[:])
    num_frameworks = len(set(results[0]))
    num_datasets = len(lines) / num_frameworks
    frameworks = results[0][::num_datasets]
    fig = plt.figure(figsize=(40, 25))
    ax = fig.add_subplot(111)
    pos = [linspace(i + 0.25, i + 0.75, num=num_frameworks, endpoint=False) for i in range(num_datasets)]
    width = pos[0][1] - pos[0][0]
    colors = ['red', 'black', 'yellow', 'blue', 'green', 'orange', 'grey']
    # Plot each dataset
    for i in range(num_datasets):
        ax.bar(pos[i],
            results[2][i::num_datasets],
            width,
            alpha=0.5,
            color=colors
            )
    if analogies:
        ax.set_ylabel('Accuracy (%)')
        ax.set_xlabel('Section')
    else:
        ax.set_ylabel('Spearman\'s Rho')
        ax.set_xlabel('Dataset')
    ax.set_xticks([0.5 + i for i in range(num_datasets)])
    ax.set_xticklabels(results[1][:num_datasets], rotation=45, fontsize='large')
    ax.set_title(title)
    # Proxy plots for adding legend correctly
    proxies = [ax.bar([0], [0], width=0, color=colors[i], alpha=0.5)[0] for i in range(num_frameworks)]
    plt.legend((proxies), frameworks, loc='best', prop={'size': 30})
    for item in [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_yticklabels():
        item.set_fontsize(30)
    # ax.get_xticklabels().set_fontsize(20)
    plt.grid()
    plt.savefig(fig_fname)
    return


if __name__ == '__main__':
    print 'Plotting reports...'
    with open(PARAMS_FILENAME, 'r') as f:
        config_str = f.readline().strip()
    plot_time_peak_mem(TIME_MEM_REPORT_FILENAME, TIME_MEM_FIG_FILENAME, config_str)
    print 'Plotted Time-Memory Report.'
    plot_word_eval_report(EVAL_WORD_VEC_REPORT_FILENAME, EVAL_WORD_VEC_FIG_FILENAME, 'Word Pairs Evaluation Report', analogies=False)
    print 'Plotted \'Word Pairs Evaluation\' Report.'
    plot_word_eval_report(EVAL_QA_REPORT_FILENAME, EVAL_QA_FIG_FILENAME, 'Analogies Task(Questions&Answers) Report', analogies=True)
    print 'Plotted \'Analogies Task(Question&Answers)\' Report.'
    print 'Finished plotting the benchmark!!!'
