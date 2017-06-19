import time
import gensim
import memory_profiler


# In[7]:

text8_file = '/home/cva/RaRe Incubator/benchmark_ml_hardware/benmark_nn_frameworks/data/text8'


# In[8]:

sentences = gensim.models.word2vec.Text8Corpus(text8_file)
print(sentences)


# In[20]:

def train_wrapper_func_for_memprof():
    start_time = time.time()
    model = gensim.models.Word2Vec(sentences, workers=6)
    print("---Gensim Train Time :  %s seconds ---" % (time.time() - start_time))


# In[21]:

print memory_profiler.memory_usage(train_wrapper_func_for_memprof, max_usage=True,multiprocess=True)