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
import re
from gensim import utils
from gensim.corpora import WikiCorpus
import snowballstemmer
stemmer = snowballstemmer.stemmer('english')

def lemmatize_wo_tags(
        content, allowed_tags=re.compile('(NN|VB|JJ|RB)'), light=False,
        stopwords=frozenset(), min_length=2, max_length=15):
    """
    This function is only available when the optional 'pattern' package is installed.
    Use the English lemmatizer from `pattern` to extract UTF8-encoded tokens in
    their base form=lemma, e.g. "are, is, being" -> "be" etc.
    This is a smarter version of stemming, taking word context into account.
    Only considers nouns, verbs, adjectives and adverbs by default (=all other lemmas are discarded).
    >>> lemmatize('Hello World! How is it going?! Nonexistentword, 21')
    ['world/NN', 'be/VB', 'go/VB', 'nonexistentword/NN']
    >>> lemmatize('The study ranks high.')
    ['study/NN', 'rank/VB', 'high/JJ']
    >>> lemmatize('The ranks study hard.')
    ['rank/NN', 'study/VB', 'hard/RB']
    """
    if not utils.has_pattern():
        raise ImportError("Pattern library is not installed. Pattern library is needed in order to use lemmatize function")
    from pattern.en import parse

    if light:
        import warnings
        warnings.warn("The light flag is no longer supported by pattern.")

    # tokenization in `pattern` is weird; it gets thrown off by non-letters,
    # producing '==relate/VBN' or '**/NN'... try to preprocess the text a little
    # FIXME this throws away all fancy parsing cues, including sentence structure,
    # abbreviations etc.
    content = (' ').join(utils.tokenize(content, lower=True, errors='ignore'))

    parsed = parse(content, lemmata=True, collapse=False)
    result = []
    for sentence in parsed:
        for token, tag, _, _, lemma in sentence:
            if min_length <= len(lemma) <= max_length and not lemma.startswith('_') and lemma not in stopwords:
                if allowed_tags.match(tag):
                    # lemma += "/" + tag[:2]
                    result.append(lemma.encode('utf8'))
    return result

utils.lemmatize = lemmatize_wo_tags


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--wikifile', help='Path to wiki dump xml', required=True)
    parser.add_argument('--corpusfile', help='Path to output corpus file', required=True)
    parser.add_argument('--words_per_line', default=1000, type=int, help='Number of words on each line. Default : 1000')
    return parser.parse_args(args)

if __name__ == '__main__':

    options = parse_args(sys.argv[1:])
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    i = 0
    wiki = WikiCorpus(options.wikifile, lemmatize=False, dictionary={})
    with open(options.corpusfile, 'w') as output:
        for text in wiki.get_texts():
            words = 0
            while words < len(text):
                output.write(' '.join(stemmer.stemWords(text[words:words + options.words_per_line])) + "\n")
                words += options.words_per_line
            i = i + 1
            # if(i == 2000):
            #     break
            if (i % 10000 == 0):
                logging.info("Saved " + str(i) + " articles")
