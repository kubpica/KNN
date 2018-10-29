import pandas as pd
import numpy as np
import math as math
import operator
from scipy.spatial.distance import euclidean
from collections import Counter

class KNN(object):
    u"Klasa implementująca algorytm kNN realizujący zadanie klasyfikacji z metryką Euklidesowską."
    def __init__(self, learnData, k):
        print("hello")
        self.k = k
        self.list = learnData

    def predict(self, testData):
        print("predict")
        xLearn = np.array(self.list.ix[:, 0:4])
        yLearn = np.array(self.list['class'])
        xTest = np.array(testData.ix[:, 0:4])
        #yTest = np.array(testData.list['class'])

        listToReturn = []

        for i in range(len(xTest)):
            distances = []
            # Compute distances
            for j in range(len(xLearn)):
                distance = euclidean(xTest[i], xLearn[j])
                distances.append([distance, j])

            # Sort distances
            distances = sorted(distances)

            # Target K nearest neighbors
            targets = []
            for j in range(self.k):
                index = distances[j][1]
                targets.append(yLearn[index])

            # Append most common target
            listToReturn.append(Counter(targets).most_common(1)[0][0])

        return listToReturn



    def score(self, list):
        print("score")