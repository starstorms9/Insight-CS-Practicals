"""
Created on Mon Mar  2 14:46:02 2020


"""

#%% Imports
import numpy as np
import os
import time
import json
import pandas as pd
import random
import inspect
import pickle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import *

import tensorflow as tf
import spacy
from tensorflow.keras.layers import Dense, InputLayer, Flatten, Reshape, BatchNormalization, GRU, LSTM, Bidirectional, Embedding, Dropout, SpatialDropout1D, Conv1D, GlobalMaxPooling1D
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences

#%% Load data
df_toxic = pd.read_csv('/home/starstorms/Insight/Insight-CS-Practicals/toxic_class/toxic_comment/train.csv')

#%% Looking at classes distribution
df_female = df_toxic[df_toxic.female >= 0.0]
df_ftoxic = df_female[df_female.target==1]

#%% Data exploration
# Get lengths of comments
lens = []
for i, row in tqdm(df_female.iterrows(), total=len(df_female)) :
    lens.append(len(row.comment_text.split()))

#%% Show comment length histogram
plt.hist(lens, bins=50)
plt.suptitle('Histogram of lengths of comments (words)')
plt.show()

#%% Config variables
cf_max_len = 150
cf_batch_size = 256
cf_vocab_size = 100000
cf_val_split = .2
cf_trunc_type = 'post'
cf_padding_type = 'post'
cf_oov_tok = '<OOV>'

#%% Training data setup
df_comments = df_female.comment_text
df_labels = df_female.target
plt.hist(df_labels)
plt.show()

df_labels[df_labels >= 0.5] = 1
df_labels[df_labels < 0.5] = 0

plt.hist(df_labels) # histogram of toxicity classifications
plt.show()
print('Percent of toxic comments {:.1f}'.format( 100 * len(df_labels[df_labels==1]) / len(df_labels) ))

#%% Spacy stuff
def get_embeddings(vocab):
        max_rank = max(lex.rank for lex in vocab if lex.has_vector)
        vectors = np.ndarray((max_rank+1, vocab.vectors_length), dtype='float32')
        for lex in vocab:
            if lex.has_vector:
                vectors[lex.rank] = lex.vector
        return vectors

nlp = spacy.load('en_core_web_md', entity=False)
embeddings = get_embeddings(nlp.vocab)
               
#%% Encoding methods
def padEnc(text) :
    texts = text if type(text) == list else [text]
    ranks = [[nlp.vocab[t].rank for t in sent.replace('.', ' . ').split(' ') if len(t)>0] for sent in texts]
    padded = pad_sequences(ranks, maxlen=cf_max_len, padding=cf_padding_type, truncating=cf_trunc_type)
    return padded

#%% Generate padded and encoded text sequences
pes = []
for i, row in tqdm(df_female.iterrows(), total=len(df_female)) :
    pes.append(padEnc(row.comment_text))

pesnp = np.stack(pes).squeeze()

#%% Get word counts
# cnt = Counter()
# for i, row in tqdm(df_female.iterrows(), total=len(df_female)) :
#     for word in row.comment_text.split() :
#         cnt[word] += 1

# savePickle('counter', cnt)

#%% Put into tf datasets
raw_dataset = tf.data.Dataset.from_tensor_slices((pesnp, df_labels))

dataset_size = len(df_labels)
trn_size = int((1-cf_val_split) * dataset_size)
val_size = int(cf_val_split * dataset_size)

shuffled_set = raw_dataset.shuffle(1000000)
trn_dataset = shuffled_set.take(trn_size)
val_dataset = shuffled_set.skip(trn_size).take(val_size)

trn_dataset = trn_dataset.batch(cf_batch_size, drop_remainder=True)
val_dataset = val_dataset.batch(cf_batch_size, drop_remainder=False)

for train_x, train_y in trn_dataset.take(1) : pass
trn_batch_num = int(trn_size / cf_batch_size)
val_batch_num = int(val_size / cf_batch_size)

#%% Model class
class Toxic(tf.keras.Model):    
    def __init__(self, learning_rate=6e-4, max_length=100, training=True, embeddings=[]):
        super(Toxic, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.training = training
        self.dropoutRate = 0.15
        self.loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
        spacy_embeddings = self.get_embeddings() if len(embeddings)==0 else embeddings
        
        model = tf.keras.Sequential()
        model.add(Embedding(spacy_embeddings.shape[0], spacy_embeddings.shape[1], input_length=max_length, trainable=False, weights=[spacy_embeddings] ) )
        model.add(SpatialDropout1D(self.dropoutRate))
        model.add(Bidirectional(GRU(128, kernel_regularizer=regularizers.l2(0.001))))
        model.add(Dropout(self.dropoutRate))
        model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(self.dropoutRate))
        model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(self.dropoutRate))
        model.add(Dense(1))
        self.model = model
    
    def __call__(self, text) :
        return self.model(text)
    
    @tf.function
    def compute_loss(self, ypred, y):
        return self.loss_func(ypred, y)
    
    @tf.function
    def trainStep(self, x, y) :
        with tf.GradientTape() as tape:
            ypred = self.model(x, training=True)
            loss = self.compute_loss(ypred, y)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))  
        return loss
    
    def saveMyModel(self, dir_path, epoch):        
        self.model.save_weights(os.path.join(dir_path, 'epoch_{}.h5'.format(epoch)))
        
    def loadMyModel(self, dir_path, epoch):        
        self.model.load_weights(os.path.join(dir_path, 'epoch_{}.h5'.format(epoch)))
        
    def setLR(self, learning_rate) :
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def printMSums(self) :
        print("Text Net Summary:\n")
        self.model.summary()

#%% Define model
txtmodel = Toxic(learning_rate=6e-4, max_length=cf_max_len, training=True, embeddings=embeddings)
txtmodel.printMSums()
txtmodel.model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

#%% Train model
history = txtmodel.model.fit(trn_dataset.repeat(), epochs=10, validation_data=val_dataset.shuffle(100).take(10), verbose=1, steps_per_epoch=trn_batch_num)

#%% Explore results










#%% Manual training methods for debugging
def getTestSetLoss(dataset, batches=0) :    
    losses = []
    for test_x, test_y in dataset.take(batches) :
        pred = txtmodel.model(test_x)
        test_loss_batch = txtmodel.compute_loss(pred, test_y)
        losses.append(test_loss_batch)
    return np.mean(losses)

def getAcc(dataset, batches=10) :    
    accs = []
    for test_x, test_y in dataset.take(batches) :
        pred = txtmodel.model(test_x).numpy().squeeze()
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1
        y_np = test_y.numpy().squeeze()
        accuracy = (y_np == pred).mean()
        accs.append(accuracy)
    return np.mean(accs)

def trainModel(epochs) :
    print('\n\nStarting training...\n')
    txtmodel.training=True
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        losses = []
        
        for train_x, train_y in trn_dataset : # tqdm(trn_dataset, total=int(dataset_size / cf_batch_size)) :
            batch_loss = txtmodel.trainStep(train_x, train_y)
            losses.append(batch_loss)
            print('Batch loss {:.3f}'.format(batch_loss))
        loss = np.mean(losses)
        
        if epoch % 1 == 0:
            test_loss = getTestSetLoss(val_dataset, 10)
            test_acc = getAcc(val_dataset, 10)
            print('   TEST LOSS: {:.1f}  TEST ACC: {:.1f}  for epoch: {}'.format(test_loss, test_acc, epoch))

        print('Epoch: {}   Train loss: {:.1f}   Epoch Time: {:.2f}'.format(epoch, float(loss), float(time.time() - start_time)))

#%% Train model
trainModel(100)