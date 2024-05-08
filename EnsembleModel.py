from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import pandas as pd

class EnsembleModel:

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._data_train = self.data_loader.data_train.select_dtypes(include=['number'])
        self._labels_train = self.data_loader.labels_train
        self._data_test = self.data_loader.data_test.select_dtypes(include=['number'])
        self._labels_test = self.data_loader.labels_test


    def VotingClassifier(self):
        lr = LogisticRegression(random_state=1)
        kn = KNeighborsClassifier(n_neighbors=5)
        svc = SVC(kernel='rbf', probability=True)

        eclf = VotingClassifier(estimators = [('lr', lr), ('kn', kn), ('svc', svc)],voting = 'hard')


        classifiers = [('Logistic Regression', lr), ('KNeighborsClassifier', kn), ('SVC', svc), ('Ensemble', eclf)]
        for label, clf in classifiers:
            scores = cross_val_score(clf, self._data_train, self._labels_train,
                                     scoring='accuracy', cv=5)
            print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))



    def GradientBoostingClassifier(self):

        gb = GradientBoostingClassifier(random_state=0)

        param_grid = dict (
            n_estimators = [5,7,9],
            learning_rate = [0.05, 0.1, 0.2],
            max_depth = [1, 3]
        )

        print("Numero de Combinações:", len(param_grid['n_estimators']) * len(param_grid['learning_rate']) * len(param_grid['max_depth']))

        grid_search = GridSearchCV(gb, param_grid=param_grid, scoring='roc_auc', cv=3)

        grid_search.fit(self._data_train, self._labels_train)

        print("Best Parameters Configuration: ", grid_search.best_params_)
        # Best Parameters Configuration:  {'learning_rate': 0.05, 'max_depth': 1, 'n_estimators': 5}

        results = pd.DataFrame(grid_search.cv_results_)
        results.sort_values(by='mean_test_score', ascending=False, inplace=True)
        results.reset_index(drop=True, inplace=True)
        print(results[['param_n_estimators','param_learning_rate','param_max_depth', 'mean_test_score', 'std_test_score']].head())
    def RandomForestClassifier(self):
        rf = RandomForestClassifier(random_state=42)

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        grid_search = GridSearchCV(rf, param_grid=param_grid, scoring='roc_auc', cv=3)
        grid_search.fit(self._data_train, self._labels_train)

        print("Best Parameters Configuration: ", grid_search.best_params_)
        results = pd.DataFrame(grid_search.cv_results_)
        results.sort_values(by='mean_test_score', ascending=False, inplace=True)
        results.reset_index(drop=True, inplace=True)
        print(results[['param_n_estimators', 'param_max_depth', 'param_min_samples_split', 'mean_test_score',
                       'std_test_score']].head())