import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.cluster import DBSCAN
from tensorflow.keras import layers, models
import seaborn as sns
from itertools import combinations

class Clustering:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._data_train = self.data_loader.data_train[['Positive_Review', 'Negative_Review']]
        self._labels_train = self.data_loader.labels_train
        self._data_test = self.data_loader.data_test[['Positive_Review', 'Negative_Review']]
        self._labels_test = self.data_loader.labels_test

    def dbscan(self):

        # Supondo que self.data_loader já esteja definido e contém data_train, labels_train, data_test, labels_test
        cols = ['Average_Score', 'Review_Total_Negative_Word_Counts', 'Review_Total_Positive_Word_Counts']
        data_train = self.data_loader.data_train.loc[:, cols]
      

        # Ajustar os parâmetros eps e min_samples
        clustering = DBSCAN(eps=2, min_samples=5).fit(data_train)
        DBSCAN_dataset = data_train.copy()
        DBSCAN_dataset['Cluster'] = clustering.labels_

        outliers = DBSCAN_dataset[DBSCAN_dataset['Cluster'] == -1]

        features = data_train.columns
        num_features = len(data_train.columns)
        combinations_list = list(combinations(features, 2))

        fig, axes = plt.subplots(num_features, num_features, figsize=(6 * num_features, 6 * num_features))

        for i in range(num_features):
            for j in range(num_features):
                if i != j:
                    feature_x, feature_y = features[i], features[j]
                    if (feature_y, feature_x) in combinations_list:
                        combinations_list.remove((feature_y, feature_x))
                    sns.scatterplot(x=feature_x, y=feature_y,
                                    data=DBSCAN_dataset[DBSCAN_dataset['Cluster'] != -1],
                                    hue='Cluster', ax=axes[i, j], palette='Set2', legend='full', s=200)
                    axes[i, j].scatter(outliers[feature_x], outliers[feature_y], s=10, label='outliers', c="k")
                    axes[i, j].legend()
                    axes[i, j].set_title(f'{feature_x} vs {feature_y}')
                    plt.setp(axes[i, j].get_legend().get_texts(), fontsize='12')
                else:
                    axes[i, j].axis('off')

        plt.tight_layout()
        plt.show()