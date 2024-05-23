import pickle

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

        self._data_train = self.data_loader.data_train.select_dtypes(include=['number']).iloc[:5000]
        self._labels_train = self.data_loader.labels_train.iloc[:5000]
        self._data_test = self.data_loader.data_test.select_dtypes(include=['number']).iloc[:5000]
        self._labels_test = self.data_loader.labels_test.iloc[:5000]

        kn = KNeighborsClassifier(n_neighbors=5)
        svc = SVC(kernel='rbf', probability=True)

        eclf = VotingClassifier(estimators = [ ('kn', kn), ('svc', svc)],voting = 'hard')

        classifiers = [('KNeighborsClassifier', kn), ('SVC', svc), ('Ensemble', eclf)]
        for label, clf in classifiers:
            clf.fit(self._data_train, self._labels_train)

            # Save the models to files using pickle
            with open('Models/'+label+'.pkl', 'wb') as dt_file:
                pickle.dump(clf, dt_file)

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

        # Save the models to files using pickle
        with open('Models/GradientBoostingClassifier.pkl', 'wb') as dt_file:
            pickle.dump(grid_search, dt_file)

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

        # Save the models to files using pickle
        with open('Models/RandomForestClassifier.pkl', 'wb') as dt_file:
            pickle.dump(grid_search, dt_file)

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

    # Está função faz um gráfico de barras com base nos resultados extraidos nos modelos:
    # 'GradientBoostingClassifier', 'RandomForestClassifier'
    def Resultados(self):
        # Resultados obtidos dos classificadores
        results = {
            'GradientBoostingClassifier': {'accuracy': 0.7293349724856574, 'precision': 0.7481432678485865,
                                           'recall': 0.73, 'f1-score': 0.72},
            'RandomForestClassifier': {'accuracy': 0.7297286749856093, 'precision': 0.7098749810193855, 'recall': 0.73,
                                       'f1-score': 0.73}
        }

        # Organize os dados para o gráfico
        classifiers = list(results.keys())
        accuracies = [results[clf]['accuracy'] for clf in classifiers]
        precisions = [results[clf]['precision'] for clf in classifiers]
        recalls = [results[clf]['recall'] for clf in classifiers]
        f1_scores = [results[clf]['f1-score'] for clf in classifiers]

        # Configuração do gráfico
        x = np.arange(len(classifiers))  # Localizações das labels
        width = 0.2  # Largura das barras

        fig, ax = plt.subplots(figsize=(12, 6))

        # Barras para cada métrica
        rects1 = ax.bar(x - 1.5 * width, accuracies, width, label='Accuracy')
        rects2 = ax.bar(x - 0.5 * width, precisions, width, label='Precision')
        rects3 = ax.bar(x + 0.5 * width, recalls, width, label='Recall')
        rects4 = ax.bar(x + 1.5 * width, f1_scores, width, label='F1-Score')

        # Adicione alguns textos e labels
        ax.set_xlabel('Classifiers')
        ax.set_ylabel('Scores')
        ax.set_title('Scores of Different Classifiers')
        ax.set_xticks(x)
        ax.set_xticklabels(classifiers)
        ax.legend()

        # Função auxiliar para adicionar valores às barras
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points de deslocamento vertical
                            textcoords="offset points",
                            ha='center', va='bottom')

        # Adicione os valores às barras
        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        autolabel(rects4)

        fig.tight_layout()

        plt.show()

    # Está função faz um gráfico de barras com base nos resultados extraidos no modelo:
    # Voting - KNeighborsClassifier, SVC, Ensemble
    def Resultados2(self):
        # Resultados obtidos dos classificadores
        results = {
            'KNeighborsClassifier': {'accuracy':0.71, 'precision': 0.52,
                                           'recall':  0.52, 'f1-score': 0.51},
            'SVC': {'accuracy': 0.73, 'precision': 0.40, 'recall': 0.43,
                                       'f1-score': 0.40},
            'Ensemble': {'accuracy': 0.72, 'precision': 0.39, 'recall': 0.45,
                    'f1-score': 0.39}
        }

        # Organize os dados para o gráfico
        classifiers = list(results.keys())
        accuracies = [results[clf]['accuracy'] for clf in classifiers]
        precisions = [results[clf]['precision'] for clf in classifiers]
        recalls = [results[clf]['recall'] for clf in classifiers]
        f1_scores = [results[clf]['f1-score'] for clf in classifiers]

        # Configuração do gráfico
        x = np.arange(len(classifiers))  # Localizações das labels
        width = 0.2  # Largura das barras

        fig, ax = plt.subplots(figsize=(12, 6))

        # Barras para cada métrica
        rects1 = ax.bar(x - 1.5 * width, accuracies, width, label='Accuracy')
        rects2 = ax.bar(x - 0.5 * width, precisions, width, label='Precision')
        rects3 = ax.bar(x + 0.5 * width, recalls, width, label='Recall')
        rects4 = ax.bar(x + 1.5 * width, f1_scores, width, label='F1-Score')

        # Adicione alguns textos e labels
        ax.set_xlabel('Classifiers')
        ax.set_ylabel('Scores')
        ax.set_title('Scores of Different Classifiers')
        ax.set_xticks(x)
        ax.set_xticklabels(classifiers)
        ax.legend()

        # Função auxiliar para adicionar valores às barras
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points de deslocamento vertical
                            textcoords="offset points",
                            ha='center', va='bottom')

        # Adicione os valores às barras
        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        autolabel(rects4)

        fig.tight_layout()

        plt.show()

        # Results
        # Accuracy: 0.71 (+/- 0.01) [KNeighborsClassifier]
        # Macro Precision: 0.52 [KNeighborsClassifier]
        # Classification Report [KNeighborsClassifier]:
        #               precision    recall  f1-score   support
        #
        #            0       0.58      0.70      0.63      2833
        #            1       0.46      0.33      0.39      2167
        #
        #     accuracy                           0.54      5000
        #    macro avg       0.52      0.52      0.51      5000
        # weighted avg       0.53      0.54      0.53      5000
        #
        # Cross-Validation Accuracy: 0.71 (+/- 0.01) [KNeighborsClassifier]
        #
        # Accuracy: 0.73 (+/- 0.01) [SVC]
        # Macro Precision: 0.40 [SVC]
        # Classification Report [SVC]:
        #               precision    recall  f1-score   support
        #
        #            0       0.52      0.71      0.60      2833
        #            1       0.28      0.15      0.20      2167
        #
        #     accuracy                           0.47      5000
        #    macro avg       0.40      0.43      0.40      5000
        # weighted avg       0.42      0.47      0.43      5000
        #
        # Cross-Validation Accuracy: 0.73 (+/- 0.01) [SVC]
        #
        # Accuracy: 0.72 (+/- 0.01) [Ensemble]
        # Macro Precision: 0.39 [Ensemble]
        # Classification Report [Ensemble]:
        #               precision    recall  f1-score   support
        #
        #            0       0.54      0.81      0.64      2833
        #            1       0.25      0.09      0.13      2167
        #
        #     accuracy                           0.49      5000
        #    macro avg       0.39      0.45      0.39      5000
        # weighted avg       0.41      0.49      0.42      5000
        #
        # Cross-Validation Accuracy: 0.72 (+/- 0.01) [Ensemble]