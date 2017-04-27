
from collections import Counter, defaultdict

import random
import numpy
from numpy import median
from sklearn.neighbors import BallTree

class Data:
    """
    Class to store embedding data
    """

    def __init__(self, location):
        import csv

        x = []
        with open(location) as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                x.append([float(entry) for entry in row])

        print("Number of embedding dimensions: {0}".format(len(x[0])))
        embedding_x = []; embedding_labels = [];
        for i, vector in enumerate(x):
            if (i + 1) >= len(x):
                break

            embedding_x.append(vector)
            embedding_labels.append(x[i + 1])

        assert len(embedding_x) == len(embedding_labels), "Did not get equal amount of predictions as points"

        fraction = 8.0/10.0
        train_set = (embedding_x[:int(fraction*len(embedding_x))], embedding_labels[:int(fraction*len(embedding_x))])
        valid_set = (embedding_x[int(fraction*len(embedding_x)):], embedding_labels[int(fraction*len(embedding_x)):])

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set


class Knearest:
    """
    kNN classifier
    """

    def __init__(self, x, y, k=5):
        """
        Creates a kNN instance

        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """

        # You can modify the constructor, but you shouldn't need to.
        # Do not use another data structure from anywhere else to
        # complete the assignment.

        self._kdtree = BallTree(x)
        self._y = y
        self._k = k

    def majority(self, item_indices):
        """
        Given the indices of training examples, return the majority label.  If
        there's a tie, return the median of the majority labels (as implemented 
        in numpy).

        :param item_indices: The indices of the k nearest neighbors
        """
        assert len(item_indices) == self._k, "Did not get k inputs"

        # Finish this function to return the most common y label for
        # the given indices.  The current return value is a placeholder 
        # and definitely needs to be changed. 
        #
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.median.html

        # Get the labels of the k nearest neighbors
        knn_labels = []
        for item_index in item_indices:
            knn_labels.append(self._y[item_index])

        # Return the average of the next label
        average_vector = []
        for i in range(len(knn_labels[0])):
            average_vector.append(numpy.average([row[i] for row in knn_labels]))

        #print("knn_labels/avg_vector", knn_labels, average_vector)
        return average_vector

    def classify(self, example):
        """
        Given an example, classify the example.

        :param example: A representation of an example in the same
        format as training data
        """

        # Finish this function to find the k closest points, query the
        # majority function, and return the predicted label.
        # Again, the current return value is a placeholder 
        # and definitely needs to be changed.

        # Use the query function on `_kdtree` to get the k closest neighbors.
        #dist, ind = self._kdtree.query(example.reshape(1, -1), k=self._k)
        dist, ind = self._kdtree.query(numpy.array(example).reshape(1, -1), k=self._k)

        return self.majority(ind[0])

    def error(self, prediction, truth, training):
        """
        Compute the accuracy of our forecat.
        """

        assert len(prediction) == len(truth), "Number of predictions much equal the number of real data points."

        error = numpy.abs(numpy.array(prediction) - numpy.array(truth))
        random_walk = numpy.abs(numpy.diff(training, axis = 0)).sum()
        return numpy.array(error/((len(prediction)/(len(training) - 1.0))*random_walk)).sum()

