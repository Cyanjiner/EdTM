# %%
import imp
import gensim
from sympy import true
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
# %%
## Import OJS text file
ojs = open("ojs_docs.txt").read().split('\n')
# %%
### Import English stop words
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
# %%
def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]
# %%
ojs_words = list(sent_to_words(ojs))
# %%
### remove stop words in ojs words list
ojs_words_nonstop = remove_stopwords(ojs_words)
# %%
import gensim.corpora as corpora
# create dictionary
id2word = corpora.Dictionary(ojs_words_nonstop)
# create corpus
texts = ojs_words_nonstop
# term document frequency
corpus = [id2word.doc2bow(text) for text in texts]
# %%
# LDA model training
from pprint import pprint
# number of topics 
num_topics = 50
lda_model = gensim.models.LdaMulticore(corpus=corpus,id2word=id2word,num_topics=num_topics)
doc_lda = lda_model[corpus]
# %%
topics = lda_model.show_topics(num_words=25)
# %%
# analyze lda with pyLDAvix
import pyLDAvis
import pickle
import os
from pyLDAvis import gensim_models
from pyLDAvis.gensim_models import prepare
pyLDAvis.enable_notebook()
LDAvis_data_filepath = os.path.join('./ojs_ldavis_prepared_'+str(num_topics))
# %%
if 1 == 1:
    LDAvis_prepared = prepare(lda_model, corpus, id2word)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)

pyLDAvis.save_html(LDAvis_prepared, './ojs_ldavis_prepared_'+ str(num_topics) +'.html')

LDAvis_prepared
# %%
