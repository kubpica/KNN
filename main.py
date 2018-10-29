import pandas as pd
import numpy as np
import kNN

names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
testData = pd.read_csv('data/iris.data.test', sep=',', header=None, names=names)
learnData = pd.read_csv('data/iris.data.learning', sep=',', header=None, names=names)

knn = kNN.KNN(learnData, 5)
xTest = np.array(testData.ix[:, 0:4])  # Get data without labels
yTest = np.array(testData['class'])  # Get only labels

print("Predicted:", knn.predict(xTest))
print("Score:", knn.score(xTest, yTest))
