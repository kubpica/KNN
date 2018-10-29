import unittest
import kNN
import numpy as np
import pandas as pd


class TestKNN(unittest.TestCase):
    def testPredict(self):
        print("testPredict")
        names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
        test_data = pd.read_csv('data/unitTest.data.test', sep=',', header=None, names=names)
        learn_data = pd.read_csv('data/unitTest.data.learning', sep=',', header=None, names=names)

        knn = kNN.KNN(learn_data, 3)
        x_test = np.array(test_data.ix[:, 0:4])  # Get data without labels

        self.assertEqual(knn.predict(x_test), ['Iris-setosa', 'Iris-setosa'])

    def testScore(self):
        print("testScore")
        names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
        test_data = pd.read_csv('data/unitTest.data.test', sep=',', header=None, names=names)
        learn_data = pd.read_csv('data/unitTest.data.learning', sep=',', header=None, names=names)

        knn = kNN.KNN(learn_data, 3)
        x_test = np.array(test_data.ix[:, 0:4])  # Get data without labels
        y_test = np.array(test_data['class'])  # Get only labels

        self.assertEqual(knn.score(x_test, y_test), 1.0)


if __name__ == '__main__':
    unittest.main()
