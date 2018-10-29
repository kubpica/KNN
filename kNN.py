import pandas as pd
import numpy as np
import math as math
import operator
from scipy.spatial.distance import euclidean
from collections import Counter


class KNN(object):
    u"Klasa implementująca algorytm kNN realizujący zadanie klasyfikacji z metryką Euklidesowską."

    def __init__(self, learn_data, k):
        print("hello")
        self.k = k
        self.data = np.array(learn_data.ix[:, 0:4])
        self.labels = np.array(learn_data['class'])

    def predict(self, x_test):
        print("predict")
        x_learn = self.data
        y_learn = self.labels

        list_to_return = []

        for i in range(len(x_test)):
            distances = []
            # Compute distances
            for j in range(len(x_learn)):
                distance = euclidean(x_test[i], x_learn[j])
                distances.append([distance, j])

            # Sort distances
            distances = sorted(distances)

            # Target K nearest neighbors
            targets = []
            for j in range(self.k):
                index = distances[j][1]
                targets.append(y_learn[index])

            # Append most common target
            list_to_return.append(Counter(targets).most_common(1)[0][0])

        return list_to_return

    def score(self, x_test):
        print("score")