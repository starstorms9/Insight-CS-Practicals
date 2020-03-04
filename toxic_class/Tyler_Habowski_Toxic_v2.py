"""
Created on Mon Mar  2 14:46:02 2020

Trained to ~95% accuracy on the validation set for the 'female' category (vs 88.4% guessing all 0's)
Tried running it separately on the 'insult' category and without any training got ~95 % (v 92% guessing all 0's)
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
import seaborn as sns
from collections import *

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

import tensorflow as tf
import spacy
from tensorflow.keras.layers import Dense, InputLayer, Flatten, Reshape, BatchNormalization, GRU, LSTM, Bidirectional, Embedding, Dropout, SpatialDropout1D, Conv1D, GlobalMaxPooling1D
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
np.set_printoptions(precision=3, suppress=True)

#%% Load data
df_toxic = pd.read_csv('/home/starstorms/Insight/Insight-CS-Practicals/toxic_class/toxic_comment/train.csv')

#%% Looking at classes distribution
df_class = df_toxic[df_toxic.insult >= 0.0]
df_ctoxic = df_class[df_class.target==1]

#%% Data exploration
# Get lengths of comments
lens = []
for i, row in tqdm(df_class.iterrows(), total=len(df_class)) :
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
df_comments = df_class.comment_text
df_labels = df_class.target
plt.hist(df_labels)
plt.show()

df_labels[df_labels >= 0.5] = 1
df_labels[df_labels < 0.5] = 0

plt.hist(df_labels) # histogram of toxicity classifications
plt.show()
print('\n\nPercent of toxic comments {:.1f}'.format( 100 * len(df_labels[df_labels==1]) / len(df_labels) ))

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

def padEnc(text) :
    texts = text if type(text) == list else [text]
    ranks = [[nlp.vocab[t].rank for t in sent.replace('.', ' . ').split(' ') if len(t)>0] for sent in texts]
    padded = pad_sequences(ranks, maxlen=cf_max_len, padding=cf_padding_type, truncating=cf_trunc_type)
    return padded

#%% Generate padded and encoded text sequences
if os.path.exists('pesnp.npy') :
    pesnp = np.load('pesnp.npy', allow_pickle=True)
else:    
    pes = []
    for i, row in tqdm(df_class.iterrows(), total=len(df_class)) :
        pes.append(padEnc(row.comment_text))
    
    pesnp = np.stack(pes).squeeze()
    np.save('pesnp_insult.npy', pesnp, allow_pickle=True)

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
    def __init__(self, learning_rate=.001, max_length=100, training=True, embeddings=[]):
        super(Toxic, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.training = training
        self.dropoutRate = 0.2
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
        model.add(Dense(1)) #, activation='sigmoid'))
        self.model = model
    
    def __call__(self, text) :
        return self.model(text).numpy().squeeze()
    
    @tf.function
    def compute_loss(self, ypred, y):
        return self.loss_func(y_true=y, y_pred=ypred)
    
    @tf.function
    def compute_loss_focused(self, ypred, y):
        return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y, tf.reshape(ypred, [-1]), 5, name=None))    
    
    @tf.function
    def trainStep(self, x, y) :
        with tf.GradientTape() as tape:
            ypred = self.model(x, training=True)
            loss = self.compute_loss(ypred, y)
            # loss = self.compute_loss_focused(ypred, y)
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

#%% 
txtmodel = Toxic(learning_rate=0.001, max_length=cf_max_len, training=True, embeddings=embeddings)
txtmodel.printMSums()

#%% Manual training methods for debugging
def getConfusionMatrix(batches_to_test=10, threshold=0.5) :
    cf = np.zeros((2,2)).astype(np.int32)
    tqdm._instances.clear()
    for train_x, train_y in tqdm(trn_dataset.take(batches_to_test), total=batches_to_test) :
        pred = txtmodel.model(train_x).numpy().squeeze()
        pred[pred < threshold] = 0
        pred[pred >= threshold] = 1
        y_np = train_y.numpy().squeeze()
        cf += confusion_matrix(y_np, pred)
    
    prec = cf[1,1] / (cf[1,1] + cf[0,1])
    reca = cf[1,1] / (cf[1,1] + cf[1,0])
    acc = (cf[0,0] + cf[1,1]) / np.sum(cf)
    return cf, prec, reca, acc

def plotCF(cf, prec, reca, acc) :
    ax = sns.heatmap(cf, annot=True, fmt="d", linewidths=1.0, xticklabels=['Predicted NO', 'Predicted YES'], yticklabels=['Actual No', 'Actual Yes'])
    ax.set_ylim([2,0])
    ax.set_title('Precision: {:.1f}  Recall: {:.1f}  Acc: {:.1f}'.format(prec*100, reca*100, acc*100))
    plt.show()
    
def plotCFAll(batches_to_test=10, threshold=0.5) :
    cf, prec, reca, acc = getConfusionMatrix(batches_to_test, threshold)
    plotCF(cf, prec, reca, acc)
    return cf, prec, reca, acc

def getTestSetLoss(dataset, batches=0) :
    losses = []
    for test_x, test_y in dataset.take(batches) :
        pred = txtmodel.model(test_x)
        test_loss_batch = txtmodel.compute_loss(pred, test_y)
        losses.append(test_loss_batch)
    return np.mean(losses)

def getAcc(dataset, batches=10, threshold=0.5) :
    accs = []
    for test_x, test_y in dataset.take(batches) :
        pred = txtmodel.model(test_x).numpy().squeeze()
        pred[pred < threshold] = 0
        pred[pred >= threshold] = 1
        y_np = test_y.numpy().squeeze()
        accuracy = (y_np == pred).mean()
        accs.append(accuracy)
    return np.mean(accs)

def trainModel(epochs) :
    tqdm._instances.clear()
    print('\n\nStarting training...\n')
    txtmodel.training=True
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        losses = []
        batch_index = 0
        
        for train_x, train_y in  tqdm(trn_dataset, total=trn_batch_num) :
            batch_loss = txtmodel.trainStep(train_x, train_y)
            losses.append(batch_loss)
            batch_index += 1
            
            if batch_index % 500 == 0 :
                print('   Batch loss so far {:.3f}   Acc: {:.3f}'.format(np.mean(losses), getAcc(val_dataset)))
                
        loss = np.mean(losses)
        
        if epoch % 1 == 0:
            test_loss = getTestSetLoss(val_dataset, 10)
            test_acc = getAcc(val_dataset, 10)
            plotCFAll(batches_to_test=50)
            print('   TEST LOSS: {:.3f}  TEST ACC: {:.1f}  for epoch: {}'.format(test_loss, 100*test_acc, epoch))

        print('Epoch: {}   Train loss: {:.3f}  Train acc: {:.1f}  Epoch Time: {:.2f}'.format(epoch, float(loss), getAcc(val_dataset)*100, float(time.time() - start_time)))

#%% Train model
trainModel(20)

#%% Plot confusion matrix and stats
cf, prec, reca, acc = plotCFAll(batches_to_test=50, threshold=0.1)
acc = getAcc(val_dataset)
f1score = 2 * (prec * reca) / (prec + reca)
print(acc, f1score)











