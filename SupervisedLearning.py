
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier

from xgboost import  XGBClassifier

class SupervisedLearning:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._data_train = self.data_loader.data_train.select_dtypes(include=['number'])
        self._labels_train = self.data_loader.labels_train
        self._data_test = self.data_loader.data_test.select_dtypes(include=['number'])
        self._labels_test = self.data_loader.labels_test

    def DecisionsTrees(self):
        # Train
        dt = DecisionTreeClassifier(max_depth=4, min_samples_split=2).fit(self._data_train, self._labels_train)
        rf = RandomForestClassifier(max_depth=9, min_samples_split=2).fit(self._data_train, self._labels_train)
        abc = AdaBoostClassifier(n_estimators=10, learning_rate=0.2).fit(self._data_train, self._labels_train)

        # Save the models to files using pickle
        with open('Models/decision_tree_model.pkl', 'wb') as dt_file:
            pickle.dump(dt, dt_file)

        with open('Models/random_forest_model.pkl', 'wb') as rf_file:
            pickle.dump(rf, rf_file)

        with open('Models/adaboost_model.pkl', 'wb') as abc_file:
            pickle.dump(abc, abc_file)

        # Prediction
        prediction_decision_tree = dt.predict(self._data_test)
        prediction_random_forest = rf.predict(self._data_test)
        prediction_adaBoost = abc.predict(self._data_test)

        # Accuracy
        accuracy_decision_tree = accuracy_score(self._labels_test, prediction_decision_tree)
        accuracy_random_forest = accuracy_score(self._labels_test, prediction_random_forest)
        accuracy_adaBoost = accuracy_score(self._labels_test, prediction_adaBoost)

        # Precision
        precision_decision_tree = precision_score(self._labels_test, prediction_decision_tree, average='macro')
        precision_random_forest = precision_score(self._labels_test, prediction_random_forest, average='macro')
        precision_adaBoost = precision_score(self._labels_test, prediction_adaBoost, average='macro')

        # Recall
        recall_decision_tree = recall_score(self._labels_test, prediction_decision_tree, average='macro')
        recall_random_forest = recall_score(self._labels_test, prediction_random_forest, average='macro')
        recall_adaBoost = recall_score(self._labels_test, prediction_adaBoost, average='macro')

        # F1
        f1_decision_tree = f1_score(self._labels_test, prediction_decision_tree, average='macro')
        f1_random_forest = recall_score(self._labels_test, prediction_random_forest, average='macro')
        f1_adaBoost = recall_score(self._labels_test, prediction_adaBoost, average='macro')

        # Print the Results
        print("\nDecision Tree:")
        print("\n  Accuracy =", accuracy_decision_tree, "| Precision =", precision_decision_tree, "| Recall =",
              recall_decision_tree, "| F1 =", f1_decision_tree)

        print("\nRandom Forest:")
        print("\n  Accuracy =", accuracy_random_forest, "| Precision =", precision_random_forest, "| Recall =",
              recall_random_forest, "| F1 =", f1_random_forest)

        print("\nAdaBoost:")
        print("\n  Accuracy =", accuracy_adaBoost, "| Precision =", precision_adaBoost, "| Recall =",
              recall_adaBoost, "| F1 =", f1_adaBoost)

        # Resultados das métricas
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        decision_tree_scores = [accuracy_decision_tree, precision_decision_tree, recall_decision_tree, f1_decision_tree]
        random_forest_scores = [accuracy_random_forest, precision_random_forest, recall_random_forest, f1_random_forest]
        adaBoost_scores = [accuracy_adaBoost, precision_adaBoost, recall_adaBoost, f1_adaBoost]

        # Configuração do gráfico
        x = np.arange(len(metrics))  # posições das métricas no eixo x
        width = 0.25  # largura das barras

        # Criação do gráfico de barras
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width, decision_tree_scores, width, label='Decision Tree')
        bars2 = ax.bar(x, random_forest_scores, width, label='Random Forest')
        bars3 = ax.bar(x + width, adaBoost_scores, width, label='AdaBoost')

        # Adicionando os rótulos e título
        ax.set_xlabel('Métricas')
        ax.set_ylabel('Scores')
        ax.set_title('Comparação das Métricas entre Decision Tree, Random Forest e AdaBoost')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()

        # Adicionando os valores das barras no topo
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate('{}'.format(round(height, 2)),
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points de deslocamento vertical
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)

        plt.tight_layout()
        plt.savefig("Images/Comparison_Metrics.png")
        plt.show()

        # Plot the Decision Tree
        plt.figure(figsize=(12, 6), dpi=300)
        plot_tree(dt)
        plt.title("Decision Tree")
        plt.savefig("Images\SupLearn_Decision_Tree.png")
        plt.show()

        # Plot a tree from Random Forest
        plt.figure(figsize=(12, 6), dpi=300)
        plot_tree(rf.estimators_[0], feature_names=self._data_train.columns, class_names=True, filled=True)
        plt.title("Random Forest - Tree 1")
        plt.savefig("Images/SupLearn_Random_Forest_Tree.png")
        plt.show()

        # Plot a tree from AdaBoost
        plt.figure(figsize=(12, 6), dpi=300)
        plot_tree(abc.estimators_[0], feature_names=self._data_train.columns, class_names=True, filled=True)
        plt.title("AdaBoost - Tree 1")
        plt.savefig("Images/SupLearn_AdaBoost_Tree.png")
        plt.show()

        # Feature Importances
        feature_names = self._data_train.columns

        # Decision Tree Feature Importances
        dt_feature_importances = dt.feature_importances_
        indices_dt = np.argsort(dt_feature_importances)[::-1]
        sorted_feature_names_dt = [feature_names[i] for i in indices_dt]

        plt.figure(figsize=(10, 6))
        plt.title('Decision Tree - Feature Importances')
        plt.bar(range(len(dt_feature_importances)), dt_feature_importances[indices_dt], align='center')
        plt.xticks(range(len(dt_feature_importances)), sorted_feature_names_dt, rotation=90)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.show()

        # Random Forest Feature Importances
        rf_feature_importances = rf.feature_importances_
        indices_rf = np.argsort(rf_feature_importances)[::-1]
        sorted_feature_names_rf = [feature_names[i] for i in indices_rf]

        plt.figure(figsize=(10, 6))
        plt.title('Random Forest - Feature Importances')
        plt.bar(range(len(rf_feature_importances)), rf_feature_importances[indices_rf], align='center')
        plt.xticks(range(len(rf_feature_importances)), sorted_feature_names_rf, rotation=90)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.show()

        # AdaBoost Feature Importances
        abc_feature_importances = abc.feature_importances_
        indices_abc = np.argsort(abc_feature_importances)[::-1]
        sorted_feature_names_abc = [feature_names[i] for i in indices_abc]

        plt.figure(figsize=(10, 6))
        plt.title('AdaBoost - Feature Importances')
        plt.bar(range(len(abc_feature_importances)), abc_feature_importances[indices_abc], align='center')
        plt.xticks(range(len(abc_feature_importances)), sorted_feature_names_abc, rotation=90)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.show()

        # Hyperparameter Tuning
        # param_dist = {'max_depth': randint(1, 10)}
        # rf2 = RandomForestClassifier()
        # rand_search = RandomizedSearchCV(rf2, param_distributions=param_dist,n_iter=3,cv=5)
        # rand_search.fit(self._data_train, self._labels_train)
        # best_rf = rand_search.best_estimator_
        # print('\nBest hyperparameters:', rand_search.best_params_)

        # For this example I only used the 'max_depth' parameter but I could use many others like n_estimators...
        # and I'm just testing for the random forest classifier just so it doesn't take too long
        # Results:
        # Best hyperparameters: {'max_depth': 9}

    def MLPClassifier(self):
        # Neural network Classifier (supervised)

        # Train the Model
        mlp = MLPClassifier(solver='adam', hidden_layer_sizes=(5, 2), max_iter=20, activation='relu')
        mlp.fit(self._data_train, self._labels_train)

        # Save the models to files using pickle
        with open('Models/MLPClassifier.pkl', 'wb') as dt_file:
            pickle.dump(mlp, dt_file)


        # Prediction
        prediction_mlp = mlp.predict(self._data_test)

        # Accuracy
        accuracy_mlp = accuracy_score(self._labels_test, prediction_mlp)

        # Precision
        precision_mlp = precision_score(self._labels_test, prediction_mlp, average='macro')


        # Print the Results
        print("\nMLP Classifier:")
        print("\n  Accuracy =", accuracy_mlp, "| Precision =", precision_mlp)


        # Plot the MLP
        plt.figure(figsize=(12, 6))
        plt.plot(mlp.loss_curve_)
        plt.title("Loss Curve", fontsize=14)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.savefig("Images\SupLearn_MLP_LossCurve.png")
        plt.show()


    def MulticlassClassifier(self):


        # Train
        oneVsRest = OneVsRestClassifier(LinearSVC(dual="auto", random_state=0))
        oneVsRest.fit(self._data_train, self._labels_train)

        oneVsOne = OneVsOneClassifier(LinearSVC(dual="auto", random_state=0))
        oneVsOne.fit(self._data_train, self._labels_train)

        # Save the models to files using pickle
        with open('Models/oneVsRest.pkl', 'wb') as dt_file:
            pickle.dump(oneVsRest, dt_file)

        with open('Models/oneVsOne.pkl', 'wb') as dt_file:
            pickle.dump(oneVsOne, dt_file)

        # Prediction with Cross Validation
        prediction_oneVsRest = oneVsOne.predict(self._data_test)
        prediction_oneVsOne = oneVsOne.predict(self._data_test)

        # Accuracy
        accuracy_oneVsRest = accuracy_score(self._labels_test, prediction_oneVsRest)
        accuracy_oneVsOne = accuracy_score(self._labels_test, prediction_oneVsOne)

        # Precision
        precision_oneVsRest = precision_score(self._labels_test, prediction_oneVsRest, average='macro')
        precision_oneVsOne = precision_score(self._labels_test, prediction_oneVsOne, average='macro')

        # Print the Results
        print("\nOneVsRest Classifier:")
        print("\n  Accuracy =", accuracy_oneVsRest, "| Precision =", precision_oneVsRest)

        print("\nOneVsOne Classifier:")
        print("\n  Accuracy =", accuracy_oneVsOne, "| Precision =", precision_oneVsOne)


    def XGBClassifier(self):

        xgb = XGBClassifier(max_depth=2, learning_rate=0.05, n_estimators=5, objective='binary:logistic', random_state=0).fit(self._data_train, self._labels_train)

        # Save the models to files using pickle
        with open('Models/XGBClassifier.pkl', 'wb') as dt_file:
            pickle.dump(xgb, dt_file)


        # Prediction
        prediction_xgb = xgb.predict(self._data_test)

        # Accuracy
        accuracy_xgb = accuracy_score(self._labels_test, prediction_xgb)

        # Precision
        precision_xgb = precision_score(self._labels_test, prediction_xgb, average='macro')

        # Print the Results
        print("\nXGBC lassifier:")
        print("\n  Accuracy =", accuracy_xgb, "| Precision =", precision_xgb)


    # Está função faz um gráfico de barras com base nos resultados extraidos nos modelos:
    # 'MLP Classifier', 'OneVsRest Classifier','OneVsOne Classifier', 'XGBClassifier'

    def Resultados(self):
        # Resultados obtidos
        results = {
            'MLP Classifier': {'accuracy': 0.5561712256604363, 'precision': 0.5378687067196164},
            'OneVsRest Classifier': {'accuracy': 0.6082061887852656, 'precision': 0.5957742570449722},
            'OneVsOne Classifier': {'accuracy': 0.6082061887852656, 'precision': 0.5957742570449722},
            'XGBClassifier': {'accuracy': 0.5692921528075857, 'precision': 0.28464607640379286}
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


        # Resultados dos Modelos
        #     Decision Tree:
        #
        #     Accuracy = 0.62513169703828 | Precision = 0.6337766957473617 | Recall = 0.583646556509994 | F1 = 0.5617739905224064
        #
        #     Random Forest:
        #
        #     Accuracy = 0.7352694423849846 | Precision = 0.7335195101653192 | Recall = 0.7208819297121775 | F1 = 0.7208819297121775
        #
        #     AdaBoost:
        #
        #     Accuracy = 0.7105006438521871 | Precision = 0.7061981310868695 | Recall = 0.7088153617747305 | F1 = 0.7088153617747305
        #
        #
        # MLPClassifier
        #
        #     MLP Classifier:
        #
        #     Accuracy = 0.5561712256604363 | Precision = 0.5378687067196164
        #
        # MulticlassClassifier
        #
        #     OneVsRes Classifier:
        #
        #     Accuracy = 0.6082061887852656 | Precision = 0.5957742570449722
        #
        #     OneVsOne Classifier:
        #
        #     Accuracy = 0.6082061887852656 | Precision = 0.5957742570449722
        #
        # XGBClassifier
        #
        #     XGB Classifier:
        #
        #     Accuracy = 0.5692921528075857 | Precision = 0.28464607640379286

