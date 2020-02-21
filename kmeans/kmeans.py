'''
Finished late, spent way too long on making it fancy compared to the actual algorithm
'''

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#%% Main Class
class kmeans() :
    def __init__(self, clusters, max_iterations=100, min_error=0.0) :
        self.clusters = clusters
        self.max_iterations = max_iterations
        self.data = None
        self.min_error = min_error
        self.centroids = []
        
    def fitOn(self, data) :
        self.data = data 
        self.labels = np.zeros((len(data)))
        self.randomPoints()
        self.iterate()
        
    def randomPoints(self) :
        mins = self.data.min(axis=0)
        maxs = self.data.max(axis=0)
        centroids = [np.random.random((2)) * abs(mins-maxs) + mins for i in range(self.clusters)]
        self.centroids = centroids
    
    def iterate(self) :
        for i in range(self.max_iterations) :
            self.assignLabels()
            self.centroids, error = self.centroidAverages()
            output_string = 'Iteration: {:2d}  Error: {:.3f}'.format(i, error)
            self.scatterAll(title=output_string)
            print(output_string)            
            if (error < self.min_error) :
                break
    
    def assignLabels(self) :
        for j in range(len(self.data)) :
                point = self.data[j]
                dists = [self.getDist(c, point) for c in self.centroids]
                self.labels[j] = np.argmin(dists)
    
    def centroidAverages(self) :
        sums = [(0,0)]*len(self.centroids)
        counts = [0.000001]*len(self.centroids)
        for i, point in enumerate(self.data) :
            label = int(self.labels[i])
            sums[label] += point
            counts[label] += 1
        sums = np.array(sums)
        counts = np.array(counts)
        new_centroids = np.divide(sums.T,counts).T 
        errors = [self.getDist(c, oldc) for c, oldc in zip(new_centroids, self.centroids)]
        error = np.mean(errors)
        return new_centroids, error
        
    def getClosest(self, point) :
        dists = []
        for p in self.data :
            dists.append( self.getDist(p,point) )
        dists = sorted(dists)
    
    def getDist(self, a, b) :
        return np.linalg.norm(a-b)
    
    def scatterAll(self, title = '') :
        datadf = pd.DataFrame(self.data)
        datadf['labels'] = self.labels
        datadf.columns = ['x', 'y', 'label']
        _ = sns.scatterplot(data=datadf, x='x', y='y', hue='label', s=10, linewidth=0)
        
        cts = pd.DataFrame(self.centroids)
        cts.columns = ['x', 'y']
        _ = sns.scatterplot(data=cts, x='x', y='y', s=30, linewidth=0)
        _ = plt.show()
    
    def scatter(self, data) :
        plt.scatter(data[:,0], data[:,1])        
    
class dataGen() :
    def __init__(self, numpoints, clusters, spread, seed=0) :
        self.numpoints = numpoints
        self.clusters = clusters
        self.spread = spread
        self.ppercluster = int(self.numpoints/self.clusters)
        self.data = self.randomData()
        if seed > 0 : np.random.seed(seed)
    
    def randomData(self):
        data = [np.random.normal(size=(self.ppercluster, 2)) + np.random.randint(-self.spread,self.spread,size=(2)) for i in range(self.clusters)]
        data  = np.concatenate(data)
        np.random.shuffle(data)
        return data

#%% Testing
km = kmeans(8, max_iterations=20, min_error=.00005)
datagenerator = dataGen(numpoints=2000, clusters=4, spread=10, seed=0)
data = datagenerator.randomData()
centroids = km.fitOn(data)
km.scatterAll()