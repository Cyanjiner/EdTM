from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from string import punctuation
from scipy import sparse

import numpy as np 
import fasttext
import pickle
import random
import math
import sys
import os

import argparse


parser = argparse.ArgumentParser(description="Text preprocessing for the Embedded Topic Model")

### Data and file related arguments
parser.add_argument("--corpus", type=str, default="corpus", help="Name of corpus")
parser.add_argument("--corpus_path", type=str, default="data/corpus.txt", help="Directory containing corpus")
parser.add_argument("--save_path", type=str, default="data", help="Directory to save results")

### Vocab and stopword arguments
parser.add_argument("--max_docfreq", type=float, default=0.7, help="Proportion of documents {0,1} in which a word must appear to be included in the vocabulary")
parser.add_argument("--min_docfreq", type=int, default=10, help="Number of documents (int) in which a word must appear to be included in the vocabulary")
parser.add_argument("--max_vocab", type=int, default=50000, help="Maximum number of words in the vocabulary")
parser.add_argument("--stopwords_path", type=str, default="stops.txt", help="Stopwords to be removed from documents")

### Vectorization arguments
parser.add_argument("--embed_dim", type=int, default=300, help="dimension of embeddings")
parser.add_argument("--load_pretrained", type=bool, default=0, help="Whether to load pretrained embeddings or train them from scratch")
parser.add_argument("--pretrained_path", type=str, default="data/corpus.bin", help="Path to fastText model with pretrained embeddings")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train fastText embeddings")
#parser.add_argument("--vectorizer", type=str, default="count", help="'count' or 'tfidf' vectorizer")

### Split arguments
parser.add_argument("--train_prop", type=float, default=0.85, help="Proportion of documents to be used for training")

args = parser.parse_args()

#%%
### Set seed for reproducibility
np.random.seed(42)

### Read corpus -- each line in the corpus file is a short document (usually a sentence)
with open(args.corpus_path, "r") as infile:
    docs = [line.lower().strip() for line in infile.readlines()]
print(f"{len(docs)} docs loaded from {args.corpus}.")

### Load stopwords (https://github.com/adjidieng/ETM/blob/b1bf69a178a9158f744930169fa821a334104256/scripts/stops.txt)
with open(args.stopwords_path, "r") as infile:
    stops = [line.strip() for line in infile.readlines()]
# Add roman numerals to stopwords
stops.extend(["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"])

### Initialize choice of vectorizer
#if args.vectorizer == "tfidf":
#    vec = TfidfVectorizer(max_df=args.max_docfreq, min_df=args.min_docfreq, max_features=args.max_vocab, tokenizer=word_tokenize, stop_words=stops)
#else:
vec = CountVectorizer(max_df=args.max_docfreq, min_df=args.min_docfreq, max_features=args.max_vocab, tokenizer=word_tokenize, stop_words=stops)

### Vectorize corpus
print("Vectorizing corpus...")
matrix = vec.fit_transform(docs) # Sparse matrix of shape (corpus_size, vocab_size)

### Get vocabulary
vocab = vec.get_feature_names_out()
print(f"{len(vocab)} word types in the corpus.")
print(f"   Initial vocabulary size: {len(vocab)}")

### Train-test-validation split
print("Splitting documents into train/test/validation sets...")
n_docs = matrix.shape[0]
test_prop = (1.0 - args.train_prop) * 2/3

ntrain = int(np.floor(args.train_prop * n_docs))
ntest = int(np.floor(test_prop * n_docs))
nval = n_docs - ntrain - ntest
idx_permute = np.random.permutation(n_docs).astype(int)

train_idx = idx_permute[: ntrain]
test_idx = idx_permute[ntrain: ntrain + ntest]
val_idx = idx_permute[ntrain + ntest:]

### Remove word types not in train data and map vocab to indices

vocab_index = matrix[train_idx,:].sum(axis=0).nonzero()[1]

vocab = sorted(
    [w for i, w in enumerate(vocab) if i in vocab_index]
    )
word2id = dict([(w, i) for i, w in enumerate(vocab)])
id2word = dict([(i, w) for i, w in enumerate(vocab)])
print(f"   Vocabulary size after removing words not in train data: {len(vocab)}")

### Save vocabulary

with open(os.path.join(args.save_path, "vocab.pkl"), "wb") as outfile:
    pickle.dump(vocab, outfile)
# Save word-to-index mapping
with open(os.path.join(args.save_path, "word2id.pkl"), "wb") as outfile:
    pickle.dump(word2id, outfile)
# Save index-to-word mapping
with open(os.path.join(args.save_path, "id2word.pkl"), "wb") as outfile:
    pickle.dump(id2word, outfile)

print(f"Vocabulary saved at {args.save_path}.")

### Represent documents as lists of word indices

docs_train = [
    [word2id[w] for w in word_tokenize(docs[idx_permute[i]]) if w in word2id] for i in train_idx
]
docs_test = [
    [word2id[w] for w in word_tokenize(docs[idx_permute[i]]) if w in word2id] for i in test_idx
]
docs_val = [
    [word2id[w] for w in word_tokenize(docs[idx_permute[i]]) if w in word2id] for i in val_idx
]
print(f"   Number of documents (train): {len(docs_train)} [this should equal {ntrain}]")
print(f"   Number of documents (test): {len(docs_test)} [this should equal {ntest}]")
print(f"   Number of documents (validation): {len(docs_val)} [this should equal {nval}]")

### Remove empty documents

print("Removing empty and one-word documents...")
docs_train = [doc for doc in docs_train if doc!=[]]
docs_test = [doc for doc in docs_test if doc!=[]]
docs_val = [doc for doc in docs_val if doc!=[]]

### Remove documents with length < 2
docs_train = [doc for doc in docs_train if len(doc) > 1]
docs_test = [doc for doc in docs_test if len(doc) > 1]
docs_val = [doc for doc in docs_val if len(doc) > 1]

### Split test set
print("Splitting test set into 2 halves...")
docs_test_h1 = [[w for i,w in enumerate(doc) if i <= (len(doc) / 2.0) - 1] for doc in docs_test]
docs_test_h2 = [[w for i,w in enumerate(doc) if i > (len(doc) / 2.0) - 1] for doc in docs_test]

if args.load_pretrained:
    print("Loading pretrained model...")
    model = fasttext.load_model(args.pretrained_path) #TODO: ensure corpus vocab and model vocab match

    embeddings = np.zeros((len(vocab), args.embed_dim))

    for word in vocab:
        embeddings[word2id[word],:] = model[word]

    print("Pre-trained word embeddings loaded.")

    with open(os.path.join(args.save_path, "embeddings.npy"), "wb") as outfile:
        np.save(outfile, embeddings)

else:
    print("Training fastText word embeddings...")
    model = fasttext.train_unsupervised(
        input=args.corpus_path, dim=args.embed_dim, epoch=args.epochs)

    embeddings = np.zeros((len(vocab), args.embed_dim))

    for word in vocab:
        embeddings[word2id[word],:] = model[word]
    
    print("Word embeddings trained.")
    with open(os.path.join(args.save_path, "embeddings.npy"), "wb") as outfile:
        np.save(outfile, embeddings)

del model, embeddings

### Get lists of words for each split
print("Creating lists of words in each split...")

def create_list_words(in_docs):
    return [x for y in in_docs for x in y]

words_train = create_list_words(docs_train)
words_test = create_list_words(docs_test)
words_test_h1 = create_list_words(docs_test_h1)
words_test_h2 = create_list_words(docs_test_h2)
words_val = create_list_words(docs_val)

print("   Total number of words in train set: ", len(words_train))
print("   Total number of words in test set: ", len(words_test))
print("   Total number of words in validation set: ", len(words_val))

### Make lists of doc indices for each split
print("Mapping lists of words to document indices...")

def create_doc_indices(in_docs):
    aux = [[j for i in range(len(doc))] for j, doc in enumerate(in_docs)]
    return [int(x) for y in aux for x in y]

doc_indices_train = create_doc_indices(docs_train)
doc_indices_test = create_doc_indices(docs_test)
doc_indices_test_h1 = create_doc_indices(docs_test_h1)
doc_indices_test_h2 = create_doc_indices(docs_test_h2)
doc_indices_val = create_doc_indices(docs_val)

print("  len(np.unique(doc_indices_train)): {} [this should be {}]".format(len(np.unique(doc_indices_train)), len(docs_train)))
print("  len(np.unique(doc_indices_test)): {} [this should be {}]".format(len(np.unique(doc_indices_test)), len(docs_test)))
print("  len(np.unique(doc_indices_test_h1)): {} [this should be {}]".format(len(np.unique(doc_indices_test_h1)), len(docs_test_h1)))
print("  len(np.unique(doc_indices_test_h2)): {} [this should be {}]".format(len(np.unique(doc_indices_test_h2)), len(docs_test_h2)))
print("  len(np.unique(doc_indices_val)): {} [this should be {}]".format(len(np.unique(doc_indices_val)), len(docs_val)))

### Number of documents in each set
n_docs_train = len(docs_train)
n_docs_test = len(docs_test)
n_docs_test_h1 = len(docs_test_h1)
n_docs_test_h2 = len(docs_test_h2)
n_docs_val = len(docs_val)

# Remove unused variables
del docs_train
del docs_test
del docs_test_h1
del docs_test_h2
del docs_val

### Create continuous bag-of-words (CBOW) representations of documents
print("Creating continuous bag-of-words (CBOW) representations of documents...")

def create_bow(doc_indices, words, n_docs, vocab_size):
    return sparse.coo_matrix(([1]*len(doc_indices),(doc_indices, words)), shape=(n_docs, len(vocab))).tocsr()

bow_train = create_bow(doc_indices_train, words_train, n_docs_train, len(vocab))
bow_test = create_bow(doc_indices_test, words_test, n_docs_test, len(vocab))
bow_test_h1 = create_bow(doc_indices_test_h1, words_test_h1, n_docs_test_h1, len(vocab))
bow_test_h2 = create_bow(doc_indices_test_h2, words_test_h2, n_docs_test_h2, len(vocab))
bow_val = create_bow(doc_indices_val, words_val, n_docs_val, len(vocab))

del words_train
del words_test
del words_test_h1
del words_test_h2
del words_val
del doc_indices_train
del doc_indices_test
del doc_indices_test_h1
del doc_indices_test_h2
del doc_indices_val

# Save bow matrices
# Train
sparse.save_npz(os.path.join(args.save_path, "bow_train.npz"), bow_train)
# Test
sparse.save_npz(os.path.join(args.save_path, "bow_test.npz"), bow_test)
# Test split 1
sparse.save_npz(os.path.join(args.save_path, "bow_test_h1.npz"), bow_test_h1)
# Test split 2
sparse.save_npz(os.path.join(args.save_path, "bow_test_h2.npz"), bow_test_h2)
# Val
sparse.save_npz(os.path.join(args.save_path, "bow_val"), bow_val)

print("*************")
print("Corpus data preprocessed.")
print("*************")