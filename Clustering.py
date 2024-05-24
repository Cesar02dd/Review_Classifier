import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
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
        cols = ['AverageScore', 'Review_Total_Negative_Word_Counts', 'Review_Total_Positive_Word_Counts']
        data_train = self.data_loader.data_train.loc[:, cols]

        eps = 2
        min_samples = 5
        chunk_size = 1000

        labels = np.full(data_train.shape[0], -1)

        for i in range(0, len(data_train), chunk_size):
            chunk = data_train[i:i + chunk_size]
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(chunk)
            labels[i:i + chunk_size] = clustering.labels

        with open('Models/DBSCAN.pkl', 'wb') as dt_file:
            pickle.dump(clustering, dt_file)

        DBSCAN_dataset = data_train.copy()
        DBSCAN_dataset['Cluster'] = labels

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

    def KMeansClustering(self, data_train, data_test, n_clusters=3):
        # Selecionar apenas colunas numéricas
        numeric_data_train = data_train.select_dtypes(include=['number'])
        numeric_data_test = data_test.select_dtypes(include=['number'])

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(numeric_data_train)

        train_clusters = kmeans.predict(numeric_data_train)
        test_clusters = kmeans.predict(numeric_data_test)

        # Adicionar rótulos de cluster ao conjunto de dados original como um novo campo 'Clustering'
        data_train['Clustering'] = train_clusters
        data_test['Clustering'] = test_clusters

        print(f"Cluster centers:\n{kmeans.cluster_centers_}")
        print(f"Train data cluster distribution:\n{pd.Series(train_clusters).value_counts()}")
        print(f"Test data cluster distribution:\n{pd.Series(test_clusters).value_counts()}")

        self._plot_clusters(numeric_data_train, train_clusters)

        return kmeans

    def _plot_clusters(self, data, clusters):
        # Reduzir os dados a três dimensões para plotagem 3D
        if data.shape[1] > 3:
            data = data.iloc[:, :3]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], c=clusters, cmap='viridis')
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)

        ax.set_xlabel(data.columns[0])
        ax.set_ylabel(data.columns[1])
        ax.set_zlabel(data.columns[2])

        plt.title('K-means Clustering')
        plt.show()