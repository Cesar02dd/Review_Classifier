import pickle

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from KNN_numpy import KNN_numpy
import pandas as pd


class KNN:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._data_train = self.data_loader.data_train.select_dtypes(include=['number'])
        self._labels_train = self.data_loader.labels_train
        self._data_test = self.data_loader.data_test.select_dtypes(include=['number'])
        self._labels_test = self.data_loader.labels_test

    def knn(self):

        print("\nKNN:")

        KNN = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2).fit(self._data_train,self._labels_train)

        # Prediction
        prediction_knn = KNN.predict(self._data_test)

        # Accuracy
        accuracy_knn = accuracy_score(self._labels_test, prediction_knn)

        # Print the Accuracy
        print("\n Accuracy =", (accuracy_knn *100.0))

        # Confusion Matrix
        print("\n Confusion Matrix \n",confusion_matrix(self._labels_test, prediction_knn))

        # Classification Report
        # print(classification_report(self._labels_test, prediction_knn))

        # Separate data into Folds
        kfold = KFold(n_splits=10, shuffle=True, random_state=2)

        # Cross Validation
        accuracy_knn_crossValidation = cross_val_score(KNN, self._data_train, self._labels_train, cv=kfold)

        # Print Mean Accuracy
        print("\n Mean Accuracy =", (accuracy_knn_crossValidation.mean() * 100.0))

    def knn_compare(self):
        print("\n\nKNN Comparison Fit")
        numpy_classifier = KNN_numpy(n_neighbors=3).fit(self._data_train.values, self._labels_train.values)
        sklearn_classifier = KNeighborsClassifier(n_neighbors=3).fit(self._data_train, self._labels_train)

        print("\n\nKNN Comparison Score")
        numpy_accuracy = numpy_classifier.score(self._data_test.values, self._labels_test.values)
        print("\n\nAccuracy Numpy =", numpy_accuracy)
        sklearn_accuracy = sklearn_classifier.score(self._data_test, self._labels_test)
        print("\n\nAccuracy Sklearn=", sklearn_accuracy)

        with open('Models/sklearn_classifier_model.pkl', 'wb') as file:
            pickle.dump(sklearn_classifier, file)

        pd.DataFrame([[numpy_accuracy, sklearn_accuracy]],
                     ['Accuracy'],
                     ['Numpy Implementation', 'Sklearn Implementation'])

