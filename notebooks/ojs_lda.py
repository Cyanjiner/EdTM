# %%
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
