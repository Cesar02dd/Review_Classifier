import numpy as np


class KNN_numpy:
    def __init__(self, n_neighbors=3):
        """
        Initializes the kNN classifier with the specified number of neighbors (k).

        Parameters:
        n_neighbors (int): The number of nearest neighbors to consider for classification. Default is 3.
        """
        self.n_neighbors = n_neighbors

    def euclidean_distance(self, x1, x2):
        """
        Computes the Euclidean distance between a test point and all training points.

        Parameters:
        x1 (array-like): The test point.
        x2 (array-like): The training points.

        Returns:
        array: The Euclidean distances between the test point and each training point.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    def fit(self, data_train, labels_train):
        """
        Fits the model using the given training data.

        Parameters:
        data_train (array-like): Training data of shape (n_samples, n_features).
        labels_train (array-like): Target values of shape (n_samples,).

        Returns:
        self: The fitted instance of the classifier.
        """
        self.X_train = data_train
        self.y_train = labels_train
        return self

    def predict(self, X):
        """
        Predicts the class labels for the provided data.

        Parameters:
        X (array-like): Data to predict of shape (n_samples, n_features).

        Returns:
        array: Predicted class labels of shape (n_samples,).
        """
        predictions = []
        for x_test in X:
            distances = self.euclidean_distance(x_test, self.X_train)
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = np.argmax(np.bincount(k_nearest_labels))
            predictions.append(most_common)
        return np.array(predictions)

    def score(self, data_test, labels_test):
        """
        Computes the accuracy of the classifier on the test data.

        Parameters:
        data_test (array-like): Test data of shape (n_samples, n_features).
        labels_test (array-like): True labels for the test data.

        Returns:
        float: The accuracy of the classifier.
        """
        y_pred = self.predict(data_test)
        accuracy = np.mean(y_pred == labels_test)
        return accuracy














# class KNN_numpy:
#     def __init__(self, X_train, y_train, n_neighbors=5, weights='uniform'):
#
#         self.X_train = X_train.values
#         self.y_train = y_train.values
#
#         self.n_neighbors = n_neighbors
#         self.weights = weights
#
#         self.n_classes = 3
#
#     def euclidian_distance(self, a, b):
#         return np.sqrt(np.sum((a - b)**2))
#
#     def neighbors(self, X_test, return_distance=False):
#
#         dist = []
#         neigh_ind = []
#
#         point_dist = [self.euclidian_distance(x_test, self.X_train) for x_test in X_test]
#
#         for row in point_dist:
#             enum_neigh = enumerate(row)
#             sorted_neigh = sorted(enum_neigh,
#                                   key=lambda x: x[1])[:self.n_neighbors]
#
#             ind_list = [tup[0] for tup in sorted_neigh]
#             dist_list = [tup[1] for tup in sorted_neigh]
#
#             dist.append(dist_list)
#             neigh_ind.append(ind_list)
#
#         if return_distance:
#             return np.array(dist), np.array(neigh_ind)
#
#         return np.array(neigh_ind)
#
#     def predict(self, X_test):
#
#         if self.weights == 'uniform':
#             neighbors = self.neighbors(X_test)
#             y_pred = np.array([
#                 np.argmax(np.bincount(self.y_train[neighbor]))
#                 for neighbor in neighbors
#             ])
#
#             return y_pred
#
#         if self.weights == 'distance':
#
#             dist, neigh_ind = self.neighbors(X_test, return_distance=True)
#
#             inv_dist = 1 / dist
#
#             mean_inv_dist = inv_dist / np.sum(inv_dist, axis=1)[:, np.newaxis]
#
#             proba = []
#
#             for i, row in enumerate(mean_inv_dist):
#
#                 row_pred = self.y_train[neigh_ind[i]]
#
#                 for k in range(self.n_classes):
#                     indices = np.where(row_pred == k)
#                     prob_ind = np.sum(row[indices])
#                     proba.append(np.array(prob_ind))
#
#             predict_proba = np.array(proba).reshape(X_test.shape[0],
#                                                     self.n_classes)
#
#             y_pred = np.array([np.argmax(item) for item in predict_proba])
#
#             return y_pred
#
#     def score(self, X_test, y_test):
#         y_pred = self.predict(X_test.values)
#
#         return float(sum(y_pred == y_test.values)) / float(len(y_test))

