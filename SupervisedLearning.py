
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

        # Plot the Decision Tree
        plt.figure(figsize=(12, 6), fontsize=14)
        plot_tree(dt)
        plt.title("Decision Tree")
        plt.savefig("Images\SupLearn_Decision_Tree.png")
        plt.show()

        # Results:
        #
        #     Decision Tree:
        #
        #         Accuracy = 0.756648650758034 | Precision = 0.36738395340401264 | Recall = 0.3636374500962513
        #
        #     Random Forest:
        #
        #          Accuracy = 0.8383348617587949 | Precision = 0.46184803869405594 | Recall = 0.35745415207967596
        #
        #     AdaBoost:
        #
        #         Accuracy = 0.8489980683303741 | Precision = 0.7196536428632716 | Recall = 0.42561055799288994

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


        # Results:
        #
        #     MLP Classifier:
        #
        #         Accuracy = 0.8302757214891107

    def MulticlassClassifier(self):


        # Train
        oneVsRest = OneVsRestClassifier(LinearSVC(dual="auto", random_state=0))
        oneVsRest.fit(self._data_train, self._labels_train)

        oneVsOne = OneVsOneClassifier(LinearSVC(dual="auto", random_state=0))
        oneVsOne.fit(self._data_train, self._labels_train)

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

        # Prediction
        prediction_xgb = xgb.predict(self._data_test)

        # Accuracy
        accuracy_xgb = accuracy_score(self._labels_test, prediction_xgb)

        # Precision
        precision_xgb = precision_score(self._labels_test, prediction_xgb, average='macro')

        # Print the Results
        print("\nXGBC lassifier:")
        print("\n  Accuracy =", accuracy_xgb, "| Precision =", precision_xgb)