import pandas as pd
import kNN

print("hi")

names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
testData = pd.read_csv('data/iris.data.test', sep=',', header=None, names=names)
learnData = pd.read_csv('data/iris.data.learning', sep=',', header=None, names=names)

knn = kNN.KNN(learnData, 5)
print(knn.predict(testData))