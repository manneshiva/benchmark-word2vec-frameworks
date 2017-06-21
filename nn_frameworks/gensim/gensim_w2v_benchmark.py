import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

text8_file = '../../data/text8'
sentences = gensim.models.word2vec.Text8Corpus(text8_file)
print(sentences)
model = gensim.models.Word2Vec(sentences, workers=6)
