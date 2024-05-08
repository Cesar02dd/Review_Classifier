from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score


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