import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.cluster import DBSCAN
from tensorflow.keras import layers, models
import seaborn as sns

class Clustering:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._data_train = self.data_loader.data_train[['Positive_Review', 'Negative_Review']]
        self._labels_train = self.data_loader.labels_train
        self._data_test = self.data_loader.data_test[['Positive_Review', 'Negative_Review']]
        self._labels_test = self.data_loader.labels_test

    def dbscan(self):

        X_train = self._data_train

        X_train = X_train[['Positive_Review', 'Negative_Review','Reviewer_Score']]
        clustering = DBSCAN(eps=3, min_samples=10).fit(X_train)
        DBSCAN_dataset = X_train.copy()
        DBSCAN_dataset.loc[:, 'Cluster'] = clustering.labels_
        DBSCAN_dataset.Cluster.value_counts().to_frame()
        outliers = DBSCAN_dataset[DBSCAN_dataset['Cluster'] == -1]

        fig2, (axes) = plt.subplots(1, 2, figsize=(12, 5))

        sns.scatterplot('Positive_Review', 'Reviewer_Score', data=DBSCAN_dataset[DBSCAN_dataset['Cluster'] != -1],
        hue='Cluster', ax=axes[0], palette='Set2', legend='full', s=200)

        sns.scatterplot('Negative_Review', 'Reviewer_Score', data=DBSCAN_dataset[DBSCAN_dataset['Cluster'] != -1],
        hue='Cluster', palette='Set2', ax=axes[1], legend='full', s=200)

        axes[0].scatter(outliers['Negative_Review'], outliers['Reviewer_Score'], s=10, label='outliers',c="k")

        axes[1].scatter(outliers['Positive_Review'], outliers['Reviewer_Score'], s=10, label='outliers', c="k")
        axes[0].legend()
        axes[1].legend()

        plt.setp(axes[0].get_legend().get_texts(), fontsize='12')
        plt.setp(axes[1].get_legend().get_texts(), fontsize='12')

        plt.show()