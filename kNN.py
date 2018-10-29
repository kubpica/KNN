import pandas as pd
import numpy as np
import math as math
import operator
from scipy.spatial.distance import euclidean

class KNN(object):
    u"Klasa implementująca algorytm kNN realizujący zadanie klasyfikacji z metryką Euklidesowską."
    def __init__(self, list, k):
        print("hello")
        self.k = k
        self.list = list

    def predict(self, list):
        print("predict")

    def score(self, list):
        print("score")