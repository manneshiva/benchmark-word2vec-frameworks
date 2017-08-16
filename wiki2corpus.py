#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Shiva Manne <manneshiva@gmail.com>

"""
Construct a corpus from a Wikipedia (or other MediaWiki-based) database dump.

"""

import argparse
import sys
import logging
from gensim.corpora import WikiCorpus


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--wikifile', help='Path to wiki dump xml', required=True)
    parser.add_argument('--corpusfile', help='Path to output corpus file', required=True)
    parser.add_argument('--words_per_line', default=1000, type=int, help='Number of words on each line. Default : 1000')
    return parser.parse_args(args)

if __name__ == '__main__':
    options = parse_args(sys.argv[1:])
    print options.wikifile
    print options.corpusfile
    print options.words_per_line

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    i = 0
    wiki = WikiCorpus(options.wikifile, lemmatize=False, dictionary={})
    with open(options.corpusfile, 'w') as output:
        for text in wiki.get_texts():
            words = 0
            while words < len(text):
                output.write(' '.join(text[words:words + options.words_per_line]) + "\n")
                words += options.words_per_line
            i = i + 1
            # if(i == 500):
            #     break
            if (i % 10000 == 0):
                logging.info("Saved " + str(i) + " articles")
