import pandas as pd
import kNN

print("hi")
testData = pd.read_csv('data/iris.data.test', sep=',', header=None)
learnData = pd.read_csv('data/iris.data.learning', sep=',', header=None)
knn = kNN.KNN(learnData, 5)
knn.predict(testData)