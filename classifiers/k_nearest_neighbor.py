from builtins import range
from builtins import object
import numpy as np
from scipy import stats


class KNearestNeighbor(object):
    """a kNN classifier with L2 distance"""

    x_train: np.ndarray
    y_train: np.ndarray

    def __init__(self) -> None:
        pass

    def train(self, x, y) -> None:
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.x_train = x
        self.y_train = y

    def predict(self, x, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances(x)
        elif num_loops == 1:
            dists = self.compute_distances(x)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(x)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, x: np.ndarray):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - x: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - distances: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """

        num_test = x.shape[0]
        num_train = self.x_train.shape[0]
        distances = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                diff = x[i] - self.x_train[j]
                distances[i, j] = np.sqrt(np.sum(diff**2))
        return distances

    def compute_distances_one_loop(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = x.shape[0]
        num_train = self.x_train.shape[0]
        distances = np.zeros((num_test, num_train))
        for i in range(num_test):
            diffs = x[i] - self.x_train
            distances[i] = np.sqrt(np.sum(diffs**2, 1))
        return distances

    def compute_distances(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """

        # shape: (5000,): one number for each image in the train set
        #   representing the sum of squared pixel components
        a = np.sum(self.x_train**2, axis=1)

        # shape: (500, 1): single vector of one number for each image in the
        #   test set representing the sum of squared pixel components.
        b = np.sum(x**2, axis=1)[:, np.newaxis]

        # a + b
        # shape: (500, 5000): the sum of the single value in each test image vector
        #   and each value in a

        # shape: (500, 5000): multiply a (500, 3072) matrix by a (3072, 5000) matrix
        #   to get a (500, 5000) matrix
        c = 2 * np.dot(x, self.x_train.T)

        return np.sqrt(a + b - c)

    def predict_labels(self, distances: np.ndarray, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = distances.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            lowest_score_indexes = np.argsort(distances[i])[0:k]
            closest = [self.y_train[idx] for idx in lowest_score_indexes]
            y_pred[i] = stats.mode(closest, keepdims=True)[0]
        return y_pred
