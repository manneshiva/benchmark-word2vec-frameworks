#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Shiva Manne <manneshiva@gmail.com>

"""
Construct a corpus from a Wikipedia (or other MediaWiki-based) database dump.

"""

import argparse
import logging
from gensim.corpora import WikiCorpus


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wikifile', help='Path to wiki dump xml', required=True)
    parser.add_argument('--corpusfile', help='Path to output corpus file', required=True)
    parser.add_argument(
        '--words_per_line', default=10000, type=int, help='Number of words on each line. Default : %(default)s'
    )
    return parser.parse_args()


if __name__ == '__main__':

    options = parse_args()
    wiki = WikiCorpus(options.wikifile, lemmatize=False)
    with open(options.corpusfile, 'w') as output:
        for i, text in enumerate(wiki.get_texts()):
            words = 0
            while words < len(text):
                output.write(' '.join(text[words:words + options.words_per_line]) + "\n")
                words += options.words_per_line
            if i % 10000 == 0:
                logger.info('Saved %d articles', i)
