import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


class EDA:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._data = self.data_loader.data.select_dtypes(include=['number'])
        self._labels = self.data_loader.labels

    def perform_eda(self):
        print("Exploratory Data Analysis (EDA) Report:")
        print("--------------------------------------")

        # Basic information of dataset
        print("\nBasic Information about the dataset:")
        print(self.data_loader.data.info())

        # Summary statistics
        print("\nSummary Statistics for normal data:")
        print(self.data_loader.data.describe())
        print("\nSummary Statistics for train data:")
        print(self.data_loader.data_train.describe())
        print("\nSummary Statistics for test data:")
        print(self.data_loader.data_test.describe())

        # Distribution analysis
        print("\nDistribution analysis")
        self.plot_distributions()

        # Correlation analysis
        print("\nCorrelation analysis")
        self.plot_correlation_heatmap()

        # Feature Importance analysis
        print("\nFeature Importance analysis")
        self.plot_feature_importance()

        # Violin plots
        print("\nViolin plots")
        self.violin_plots()

    def plot_distributions(self):
        """
        Plots distributions of the features.
        """
        num_cols = len(self._data.columns)
        fig, axes = plt.subplots(num_cols, 1, figsize=(10, 5 * num_cols))
        for i, feature in enumerate(self._data.columns):
            sns.histplot(data=self._data, x=feature, ax=axes[i])
            axes[i].set_title(f"Distribution of {feature}")
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self):
        """
        Plots a correlation heatmap between features and labels.
        """
        # Concatenate features and labels horizontally for correlation calculation
        data_with_labels = pd.concat([self._data, self._labels], axis=1)

        # Compute the correlation matrix
        corr_matrix = data_with_labels.corr()

        # Plot the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap between Features and Labels")
        plt.show()

    def plot_feature_importance(self, n_estimators=5, n_repeats=2):
        """
        Computes and visualizes feature importance using permutation importance.
        """
        # Fit a random forest classifier to compute feature importance
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(self._data, self._labels)

        # Compute permutation importance
        result = permutation_importance(clf, self._data, self._labels,
                                        n_repeats=n_repeats)
        sorted_idx = result.importances_mean.argsort()

        # Plot feature importance
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), self._data.columns[sorted_idx])
        plt.xlabel('Permutation Importance')
        plt.title('Feature Importance')
        plt.show()

    def violin_plots(self):
        print("TO DO: violin_plots")
