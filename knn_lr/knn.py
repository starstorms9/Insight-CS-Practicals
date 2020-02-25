"""
Started at 14:10
"""

'''
Save file as: FIRSTNAME_LASTINITIAL_KNN.py
Timestamp: 02042020
Objective: Coding KNN from scratch
Dependcies: Python 3.6, numpy

FIRST 45 MINUTES
@author: Amber

Implement k-Nearest Neighbors (kNN) in Python from scratch, you may ONLY:
import csv
import random
import math
import operator
import pandas

Use the Iris dataset + your classifier to achieve at least 90% accuracy.
The Iris Flower Dataset involves predicting the flower species given measurements of iris flowers.
It is a multiclass classification problem. The number of observations for each class is balanced.
There are 150 observations with 4 input variables and 1 output variable.
The variable names are as follows:
Sepal length in cm.
Sepal width in cm.
Petal length in cm.
Petal width in cm.
Remember kNN is supervised, so you are calculating distances from points in your test set to  ‘k’ nearest neighbors in your training set.

Functional Steps:
function 0: Load a CSV file (file found in folder)
with open('iris.data', 'rb') as csvfile:
    lines = csv.reader(csvfile)
    for row in lines:
        print ', '.join(row)
function 1: Convert string column to integer (already should be done for you!)
function 2: Find the min and max values for each column
function 3: Rescale dataset columns to the range 0-1

function 4: Split a dataset into k-folds (implement this if there is time at the end)
function 5: Calculate accuracy percentage
function 6: Evaluate an algorithm using a cross validation split
function 7: Calculate the Euclidean distance between two vectors
function 8: Locate the most similar neighbors
function9: Make a prediction with neighbors
function10: kNN Algorithm

then...

Test the kNN on the Iris Flowers dataset

print('Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

'''
#%% Imports
import math
import pandas as pd

#%% Solution
class KNN() :
    def __init__(self, data) :
        self.data = data
        self.data = self.processData(data)
        self.splitData(data, .8)
    
    def processData(self, data) :
        for col in data.columns[:-1] :
            data[col] = (data[col] - data[col].min())
            data[col] = data[col] / data[col].max()
        return data
    
    def splitData(self, data, trn_percent) :
        trnNum = int(len(data) * trn_percent)
        self.trnData = data.sample(trnNum)
        self.valData = data.drop(self.trnData.index)
    
    def predict(self, k, data, point) :
        dists = []
        for i, row in data.iterrows() :
            a = row.values[:4]
            b = point.values[:4]
            dist = self.dist(a,b)
            dists.append(dist)
        data['dist'] = dists
        
        data = data.sort_values(by=['dist'])
        closest = data[:k].label.value_counts()
        result = closest.argmax()
        return result
        
    def dist(self, a, b) :
        total = 0
        for i in range(len(a)) :
            total += math.pow(a[i]-b[i], 2)
        math.sqrt(total)
        return total

def loadData() :
    # 0 setosa	1 versicolor 2 virginica
    data = pd.read_csv('iris.csv')
    data.columns = ['seplen', 'sepwid', 'petlen', 'petwid', 'label']
    return data

#%% Testing
data = loadData()
knn = KNN(data)

correct = 0
for i in range(len(knn.valData)) :
    result = knn.predict(5, knn.trnData, knn.valData.iloc[i])
    if result == knn.valData.iloc[i].label :
        correct += 1
    
print('Accuracy: %.3f%%' % (correct / len(knn.valData)))