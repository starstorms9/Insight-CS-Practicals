"""
Created on Wed Mar  4 14:09:09 2020

Ended up having way too much fun with this one... tried many methods just to see the differences
"""

'''
Objective: Sentiment Analysis with the IMBD dataset

Sentiment analysis

Step 0: Download data from: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
Step 1: EDA (~15 minutes) 
Step 2: Data Preprocessing (~20 minutes)
- remove stop words, punctuation, capital letters
- Tokenize
- Stemming
- Lemmatizing

Step 3: Vectorization/Experimentation (~30 minutes)
- n-grams
- embeddings
- one hot encoding
- TFIDF

Step 4: Build Classifiers (~20 minutes)
- LR
- SVM

Step 5: Bonus (any additional time)
- NNs (maybe just a shallow one in keras (~10 minutes))
- BERT

Step 6: Evaluate model (~5 minutes)
- Validate
- Chose model
- Chose metric (classification_report)
'''

#%% Imports
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import re
import string
import numpy as np
from collections import *
import seaborn as sns
from scipy import spatial
import time

import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer, Flatten, Reshape, BatchNormalization, GRU, LSTM, Bidirectional, Embedding, Dropout, SpatialDropout1D, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
np.set_printoptions(precision=3, suppress=True)

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.manifold import TSNE

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 8)

#%% Load data
dfimdb = pd.read_csv('/home/starstorms/Insight/Insight-CS-Practicals/sentiment/IMDB Dataset.csv')

dfimdb.sentiment[dfimdb.sentiment == 'positive'] = 1
dfimdb.sentiment[dfimdb.sentiment == 'negative'] = 0

dfneg = dfimdb.review[dfimdb.sentiment == 0]
dfpos = dfimdb.review[dfimdb.sentiment == 1]

#%% EDA
plt.hist(dfimdb.sentiment, label='Negative vs Positive Reviews')

lens = []
cnt = Counter()

cntneg = Counter()
cntpos = Counter()

print('\n\nGetting word counts and lengths... ')
for i, row in tqdm(dfimdb.iterrows(), total=len(dfimdb)) :
    words = row.review.split()
    lens.append(len(words))
    for word in words :
        cnt[word] += 1
        if (row.sentiment == 0) :
            cntneg[word] += 1
        else :
            cntpos[word] += 1

#%% Show word counts
plt.hist(lens, bins=30)
plt.show()

num_common = 100
cnt_common = cnt.most_common(num_common)
cnt_neg_common = cntneg.most_common(num_common)
cnt_pos_common = cntpos.most_common(num_common)

cnt_common_vals = [entry[1] for entry in cnt_common]
cnt_common_words = [entry[0] for entry in cnt_common]

cnt_neg_common_vals = [entry[1] for entry in cnt_neg_common]
cnt_neg_common_words = [entry[0] for entry in cnt_neg_common]

cnt_pos_common_vals = [entry[1] for entry in cnt_pos_common]
cnt_pos_common_words = [entry[0] for entry in cnt_pos_common]

def plotCnts(words, vals) :
    plt.figure(figsize=(25,6))
    plt.bar(words, vals)
    plt.xticks(rotation='vertical')
    plt.show()
    
plotCnts(cnt_common_words, cnt_common_vals)
plotCnts(cnt_neg_common_words, cnt_neg_common_vals)
plotCnts(cnt_pos_common_words, cnt_pos_common_vals)

#%% Setup configs
cf_max_len = 500
cf_batch_size = 256
cf_val_split = .2
cf_trunc_type = 'post'
cf_padding_type = 'post'
cf_oov_tok = '<OOV>'

val_samples = int(cf_val_split * len(dfimdb))
trn_samples = len(dfimdb) - val_samples

#%% Train test split
dftrn = dfimdb.sample(trn_samples)
dfval = dfimdb.drop(dftrn.index)

#%% Spacy Setup
print('\n\nGetting spacy setup... ')
def get_embeddings(vocab):
        max_rank = max(lex.rank for lex in vocab if lex.has_vector)
        vectors = np.ndarray((max_rank+1, vocab.vectors_length), dtype='float32')
        for lex in vocab:
            if lex.has_vector:
                vectors[lex.rank] = lex.vector
        return vectors

nlp = spacy.load('en_core_web_md', entity=False)
spacy_embeddings = get_embeddings(nlp.vocab)

# Method to quickly pad and encode text sequences uses SpaCy vocab
def padEnc(text) :
    texts = text if type(text) == list else [text]
    ranks = [[nlp.vocab[t].rank for t in sent.replace('.', ' . ').split(' ') if len(t)>0] for sent in texts]
    padded = pad_sequences(ranks, maxlen=cf_max_len, padding=cf_padding_type, truncating=cf_trunc_type)
    return padded

# Method to get vectors from word tokens
def getVects(text) :
    toks = text.split() 
    rvects = np.zeros((len(toks), spacy_embeddings.shape[1]))
    for j, tok in enumerate(toks) :
        rvects[j] = nlp.vocab[tok].vector
    return rvects

thresh = lambda val : 0 if val < 0.5 else 1

#%% Create simple model based on pooling word embeddings
dftrn_neg = dftrn.review[dfimdb.sentiment == 0]
dftrn_pos = dftrn.review[dfimdb.sentiment == 1]
dfval_neg = dfval.review[dfimdb.sentiment == 0]
dfval_pos = dfval.review[dfimdb.sentiment == 1]

neg_vects_max = np.zeros((len(dftrn_neg), spacy_embeddings.shape[1]))
pos_vects_max = np.zeros((len(dftrn_pos), spacy_embeddings.shape[1]))

neg_vects_avg = np.zeros((len(dftrn_neg), spacy_embeddings.shape[1]))
pos_vects_avg = np.zeros((len(dftrn_pos), spacy_embeddings.shape[1]))

neg_vects_abm = np.zeros((len(dftrn_neg), spacy_embeddings.shape[1]))
pos_vects_abm = np.zeros((len(dftrn_pos), spacy_embeddings.shape[1]))

val_neg_vects_max = np.zeros((len(dfval_neg), spacy_embeddings.shape[1]))
val_pos_vects_max = np.zeros((len(dfval_pos), spacy_embeddings.shape[1]))

val_neg_vects_avg = np.zeros((len(dfval_neg), spacy_embeddings.shape[1]))
val_pos_vects_avg = np.zeros((len(dfval_pos), spacy_embeddings.shape[1]))

val_neg_vects_abm = np.zeros((len(dfval_neg), spacy_embeddings.shape[1]))
val_pos_vects_abm = np.zeros((len(dfval_pos), spacy_embeddings.shape[1]))

#%% Pool data according to various basic pooling functions
print('\n\nPooling word embedding data... ')
pos_index = 0
neg_index = 0
for i, row in tqdm(dftrn.iterrows(), total=len(dftrn)) :    
    rvects = getVects(row.review)
    
    if row.sentiment == 0 :
        neg_vects_max[neg_index] = np.max(rvects, axis=0)
        neg_vects_avg[neg_index] = np.mean(rvects, axis=0)
        neg_vects_abm[neg_index] = np.max(np.abs(rvects), axis=0)
        neg_index += 1
    else :
        pos_vects_max[pos_index] = np.max(rvects, axis=0)
        pos_vects_avg[pos_index] = np.mean(rvects, axis=0)
        pos_vects_abm[pos_index] = np.max(np.abs(rvects), axis=0)
        pos_index += 1
        
pos_index = 0
neg_index = 0
for i, row in tqdm(dfval.iterrows(), total=len(dfval)) :    
    rvects = getVects(row.review)
    
    if row.sentiment == 0 :
        val_neg_vects_max[neg_index] = np.max(rvects, axis=0)
        val_neg_vects_avg[neg_index] = np.mean(rvects, axis=0)
        val_neg_vects_abm[neg_index] = np.max(np.abs(rvects), axis=0)
        neg_index += 1
    else :
        val_pos_vects_max[pos_index] = np.max(rvects, axis=0)
        val_pos_vects_avg[pos_index] = np.mean(rvects, axis=0)
        val_pos_vects_abm[pos_index] = np.max(np.abs(rvects), axis=0)
        pos_index += 1

#%% Generate TSNE clusters on sample sets
def sample2D(arr, size) :
    return arr[np.random.randint(arr.shape[0], size=size), :]
        
sample_vects = 1000
x_avg = np.concatenate([ sample2D(neg_vects_avg, sample_vects), sample2D(pos_vects_avg, sample_vects)])
x_max = np.concatenate([ sample2D(neg_vects_max, sample_vects), sample2D(pos_vects_max, sample_vects)])
x_abm = np.concatenate([ sample2D(neg_vects_abm, sample_vects), sample2D(pos_vects_abm, sample_vects)])

def getLatent(vects, perp=40, lr = 200, n_iter=1000) :
    tsne = TSNE(n_components=2, n_iter=n_iter, verbose=3, perplexity=perp, learning_rate=lr)
    lvects = tsne.fit_transform(vects)
    return lvects

print('\n\nGetting latent vectors for TSNE mapping... ')
lvects_avg = getLatent(x_avg)
lvects_max = getLatent(x_max)
lvects_abm = getLatent(x_abm)

#%% Plot TSNE
def plotTSNE(tsne_vects, split_point, title='') :
    dftsne = pd.DataFrame(columns = ['tsne1', 'tsne2'])
    dftsne['tsne1'] = tsne_vects[:,0]
    dftsne['tsne2'] = tsne_vects[:,1]
    dftsne['label'] = 0
    dftsne.label[dftsne.index > split_point] = 1
    sns.scatterplot(data=dftsne, x='tsne1', y='tsne2', hue='label', s=30, linewidth=0, palette='bright')
    plt.axis('off')
    if not title=='' : plt.suptitle('TSNE for {}'.format(title))
    plt.show()

plotTSNE(lvects_avg, sample_vects, title='Average')
plotTSNE(lvects_max, sample_vects, title='Max')
plotTSNE(lvects_abm, sample_vects, title='Absolute Max')
  
#%% Get vector data
vects_max = np.concatenate([neg_vects_max, pos_vects_max])
labels = np.zeros((len(vects_max)))
labels[len(neg_vects_max):] = 1

val_vects_max = np.concatenate([val_neg_vects_max, val_pos_vects_max])
val_labels = np.zeros((len(val_vects_max)))
val_labels[len(val_neg_vects_max):] = 1

#%% Basic KNN for classification
pos_split_index = len(neg_vects_avg)
vec_tree = spatial.KDTree(np.concatenate([neg_vects_avg, pos_vects_avg]))

#%% Build KD tree for checking
print('\n\nBuilding simple KNN model... ')
num_sample_vects = 2000
x = np.concatenate([ sample2D(neg_vects_max, sample_vects), sample2D(pos_vects_max, sample_vects)])
vec_tree = spatial.KDTree(x)

def checkReview(text, k) :
    rvects = getVects(text)
    embedded = np.max(rvects, axis=0)
    n_dists, close_ids = vec_tree.query(embedded, k = k)
    sents = [0 if cid < num_sample_vects else 1 for cid in close_ids]
    sent = np.average(sents)
    return sent

print(checkReview('I love this movie.', 10))

#%% Evaluate KNN model
print('\n\nEvaluating KNN model... ')
preds = []
count = 0
for i, row in dfval.iterrows() :
    pred = checkReview(row.review, 3)
    preds.append(pred)
    if count % 50 == 0 :
        preds_thresholded = [0 if p < 0.9 else 1 for p in preds]
        acc = np.mean(dfval.sentiment[:(len(preds_thresholded))] == preds_thresholded)
        print('{} : Acc: {:.3f}'.format(i, acc))
    if count > 1000 : break
    count += 1

preds_thresholded = [0 if p < 0.9 else 1 for p in preds]
acc = np.mean(dfval.sentiment[:(len(preds_thresholded))] == preds_thresholded)
print('Accuracy: {:.3f}'.format(acc))
# ~55%, barely beating random

#%% Build simple logistic regression model on top of embeddings
clf = LogisticRegression(random_state=0, verbose=3, max_iter=10000)
clf.fit(vects_max, labels)

y_pred = clf.predict(val_vects_max)
print(classification_report(list(val_labels), list(y_pred)))
# ~72% accuracy, okay for simple model

#%% Simple Keras model to use on top of embeddings
def makeSimpleModel() :
    model = tf.keras.Sequential()
    model.add(Dense(64, input_dim=spacy_embeddings.shape[1], activation='relu'))
    model.add(Dense(1))
    
    model.summary()
    model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

dense_model = makeSimpleModel()

#%% Train and evaluate simple dense model
history = dense_model.fit(vects_max, labels, epochs=20)
preds = dense_model.predict(val_vects_max)
p = np.apply_along_axis(thresh, arr=preds, axis=1)
report = classification_report(list(val_labels), list(p))
print(report)
# Stil only ~72% accuracy, adding layers instead of logistic regression didn't seem to help




#%% Generate padded and encoded text sequences
def makePaddedEncodedText(df, name) :
    if os.path.exists(name) :
        pesnp = np.load(name, allow_pickle=True)
    else:    
        pes = []
        labs = []
        for i, row in tqdm(df.iterrows(), total=len(df)) :
            pes.append(padEnc(row.review))
            labs.append(row.sentiment)
        
        pesnp = np.stack(pes).squeeze()
        lnp = np.stack(labs).squeeze()
        np.save(name, pesnp, allow_pickle=True)
    return pesnp, lnp
    
trn_enctext, trn_labels = makePaddedEncodedText(dftrn, 'trn_enctext.npy')
val_enctext, val_labels = makePaddedEncodedText(dfval, 'val_enctext.npy')

#%% Build simple embedding model using new embeddings
def makeNewEmbedModel(embeddings, embed_dim) :
    model = tf.keras.Sequential()    
    model.add(Embedding( embeddings.shape[0], embed_dim, input_length=cf_max_len, trainable=True))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1))
    
    model.summary()
    model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

embed_model = makeNewEmbedModel(spacy_embeddings, embed_dim=16)

#%% Train and evaluate simple embedding model
history = embed_model.fit(trn_enctext, trn_labels, epochs=10)
preds = embed_model.predict(val_enctext)
p = np.apply_along_axis(thresh, arr=preds, axis=1)
print(classification_report(list(val_labels), list(p)))
# ~90% accuracy, decent for how simple the model is


#%% Build simple embedding model using SpaCy embeddings
def makeSpacyEmbedModel(embeddings) :
    model = tf.keras.Sequential()    
    model.add(Embedding( embeddings.shape[0], embeddings.shape[1], input_length=cf_max_len, trainable=False, weights=[embeddings]))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1))
    
    model.summary()
    model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

spacy_model = makeSpacyEmbedModel(spacy_embeddings)

#%% Train and evaluate simple embedding model
spacy_model.fit(trn_enctext, trn_labels, epochs=10)
preds = spacy_model.predict(val_enctext)
p = np.apply_along_axis(thresh, arr=preds, axis=1)
print(classification_report(list(val_labels), list(p)))
# Only ~70%, similar to previous which is sensible since it's basically the same thing



#%% Build GRU model using SpaCy embeddings
def makeGRUModel(embeddings, dropoutRate = 0.2) :
    model = tf.keras.Sequential()    
    model.add(Embedding( embeddings.shape[0], embeddings.shape[1], input_length=cf_max_len, trainable=False, weights=[embeddings]))
    model.add(SpatialDropout1D(dropoutRate))
    model.add(Bidirectional(GRU(32, kernel_regularizer=regularizers.l2(0.001))))
    model.add(Dropout(dropoutRate))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(dropoutRate))
    model.add(Dense(1))
    
    model.summary()
    model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

gru_model = makeGRUModel(spacy_embeddings)

#%% Train and evaluate simple embedding model
gru_model.fit(trn_enctext, trn_labels, epochs=10)
preds = gru_model.predict(val_enctext)
p = np.apply_along_axis(thresh, arr=preds, axis=1)
print(classification_report(list(val_labels), list(p)))
# ~90 % on validation data


#%% Build GRU model NOT using SpaCy embeddings and instead training embeds from scratch
def makeGRUNoSpacyModel(embeddings, embed_dim=32, dropoutRate = 0.2) :
    model = tf.keras.Sequential()    
    model.add(Embedding( embeddings.shape[0], embed_dim, input_length=cf_max_len, trainable=True))
    model.add(SpatialDropout1D(dropoutRate))
    model.add(Bidirectional(GRU(32, kernel_regularizer=regularizers.l2(0.001))))
    model.add(Dropout(dropoutRate))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(dropoutRate))
    model.add(Dense(1))
    
    model.summary()
    model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

gru_model_no_spacy = makeGRUNoSpacyModel(spacy_embeddings, 32)

#%% Train and evaluate simple embedding model
history = gru_model_no_spacy.fit(trn_enctext, trn_labels, epochs=10)
preds = gru_model_no_spacy.predict(val_enctext)
p = np.apply_along_axis(thresh, arr=preds, axis=1)
print(classification_report(list(val_labels), list(p)))
# Gets to ~96 % accuracy on training but only 87% on testing, overfitting a bit

#%% Test the final model
test_text = ['this movie was the worst ever',
             'i enjoyed this movie but overall it was pretty bad. I would no reccomend it.',
             'this movie was jam packed with hilarious things and stuff.',
             'definitely an awesome movie.',
             'the composition of the storyline was interesting and the characters were shallow at times but mostly comical.',
             'watching this movie reminded me of my childhood for some reason, brings back interesting memories',
             'not too bad but I did not really enjoy it. On the other hand, it was pretty awesome but still would not reccomend it unless you really like this kind of movie']

encoded = [padEnc(t) for t in test_text]
encoded = np.stack(encoded).squeeze()
preds = gru_model_no_spacy.predict(encoded)
preds = [thresh(p) for p in preds]

for i in range(len(preds)) :
    print('{} : {}'.format(preds[i], test_text[i]))