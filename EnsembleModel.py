import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, classification_report
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

        kn = KNeighborsClassifier(n_neighbors=5)
        svc = SVC(kernel='rbf', probability=True)

        eclf = VotingClassifier(estimators = [ ('kn', kn), ('svc', svc)],voting = 'hard')

        classifiers = [('KNeighborsClassifier', kn), ('SVC', svc), ('Ensemble', eclf)]
        for label, clf in classifiers:
            clf.fit(self._data_train, self._labels_train)
            y_pred = clf.predict(self._data_test)

            scores = cross_val_score(clf, self._data_train, self._labels_train,scoring='accuracy', cv=5)
            print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

            # Calcula a precisão macro
            precision = precision_score(self._labels_test, y_pred, average='macro')
            print("Macro Precision: %0.2f [%s]" % (precision, label))

            # Imprime o relatório de classificação
            print("Classification Report [%s]:\n%s" % (label, classification_report(self._labels_test, y_pred, zero_division=1)))

            # Cross-validation scores
            scores = cross_val_score(clf, self._data_train, self._labels_train, scoring='accuracy', cv=5)
            print("Cross-Validation Accuracy: %0.2f (+/- %0.2f) [%s]\n" % (scores.mean(), scores.std(), label))

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
        # Pega o melhor modelo
        best_model = grid_search.best_estimator_

        # Faz as previsões no conjunto de teste
        y_pred = best_model.predict(self._data_test)

        # Calcula a acurácia
        accuracy = accuracy_score(self._labels_test, y_pred)
        print("Accuracy: ", accuracy)

        # Calcula a precisão
        precision = precision_score(self._labels_test, y_pred,average='binary')
        print("Precision: ", precision)

        # Para uma visão mais completa, você pode imprimir o relatório de classificação
        print("Classification Report:\n", classification_report(self._labels_test, y_pred))

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

        # Pega o melhor modelo
        best_model = grid_search.best_estimator_

        # Faz as previsões no conjunto de teste
        y_pred = best_model.predict(self._data_test)

        # Calcula a acurácia
        accuracy = accuracy_score(self._labels_test, y_pred)
        print("Accuracy: ", accuracy)

        # Calcula a precisão
        precision = precision_score(self._labels_test, y_pred, average='binary')
        print("Precision: ", precision)

        # Para uma visão mais completa, você pode imprimir o relatório de classificação
        print("Classification Report:\n", classification_report(self._labels_test, y_pred))

    def Resultados(self):
        # Resultados obtidos dos classificadores
        results = {
            'GradientBoostingClassifier': {'accuracy': 0.7335870042441095, 'precision': 0.7708891399064063},
            'RandomForestClassifier': {'accuracy': 0 , 'precision': 0}
        }

        # Organize os dados para o gráfico
        classifiers = list(results.keys())
        accuracies = [results[clf]['accuracy'] for clf in classifiers]
        precisions = [results[clf]['precision'] for clf in classifiers]

        # Configuração do gráfico
        x = np.arange(len(classifiers))  # Localizações das labels
        width = 0.35  # Largura das barras

        fig, ax = plt.subplots(figsize=(12, 6))

        # Barras para acurácia e precisão
        rects1 = ax.bar(x - width / 2, accuracies, width, label='Accuracy')
        rects2 = ax.bar(x + width / 2, precisions, width, label='Precision')

        # Adicione alguns textos e labels
        ax.set_xlabel('Classifiers')
        ax.set_ylabel('Scores')
        ax.set_title('Accuracy and Precision of Different Classifiers')
        ax.set_xticks(x)
        ax.set_xticklabels(classifiers)
        ax.legend()

        # Função auxiliar para adicionar valores às barras
        def autolabel2(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points de deslocamento vertical
                            textcoords="offset points",
                            ha='center', va='bottom')

        # Adicione os valores às barras
        autolabel2(rects1)
        autolabel2(rects2)

        fig.tight_layout()

        plt.show()