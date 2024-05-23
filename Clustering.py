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
        # Lê o arquivo CSV contendo as avaliações de hotéis
        df = pd.read_csv('Hotel_Reviews.csv')

        # Remove linhas com valores ausentes (NA)
        df.dropna(inplace=True)

        # Seleciona as primeiras 5000 linhas e as colunas especificadas como conjunto de treino
        X_train = df.loc[:4999,
                  ['Average_Score', 'Review_Total_Negative_Word_Counts', 'Review_Total_Positive_Word_Counts']]

        # Aplica o algoritmo de clustering DBSCAN ao conjunto de treino
        clustering = DBSCAN(eps=10, min_samples=5).fit(X_train)


        # Save the models to files using pickle
        with open('Models/DBSCAN.pkl', 'wb') as dt_file:
            pickle.dump(clustering, dt_file)


        # Cria uma cópia do conjunto de treino e adiciona a coluna 'Cluster' com os rótulos do DBSCAN
        DBSCAN_dataset = X_train.copy()
        DBSCAN_dataset['Cluster'] = clustering.labels_

        # Identifica os outliers (pontos com rótulo -1)
        outliers = DBSCAN_dataset[DBSCAN_dataset['Cluster'] == -1]

        # Extrai as características usadas para o clustering
        features = X_train.columns
        num_features = len(X_train.columns)

        # Gera todas as combinações possíveis de duas características
        combinations_list = list(combinations(features, 2))

        # Cria uma grade de subplots para visualizar as combinações das características
        fig, axes = plt.subplots(num_features, num_features, figsize=(6 * num_features, 6 * num_features))

        # Itera sobre todas as características para preencher a grade de subplots
        for i in range(num_features):
            for j in range(num_features):
                if i != j:
                    # Seleciona a combinação de características atual
                    feature_x, feature_y = features[i], features[j]
                    # Remove combinações duplicadas
                    if (feature_y, feature_x) in combinations_list:
                        combinations_list.remove((feature_y, feature_x))
                    # Plota os dados do clustering, colorindo por rótulo de cluster
                    sns.scatterplot(x=feature_x, y=feature_y,
                                    data=DBSCAN_dataset[DBSCAN_dataset['Cluster'] != -1],
                                    hue='Cluster', ax=axes[i, j], palette='Set2', legend='full', s=200)
                    # Plota os outliers como pontos pretos
                    axes[i, j].scatter(outliers[feature_x], outliers[feature_y], s=10, label='outliers', c="k")
                    axes[i, j].legend()
                    axes[i, j].set_title(f'{feature_x} vs {feature_y}')
                    plt.setp(axes[i, j].get_legend().get_texts(), fontsize='12')
                else:
                    # Desativa o subplot para a diagonal principal
                    axes[i, j].axis('off')

        # Ajusta o layout dos subplots para evitar sobreposição
        plt.tight_layout()
        # Exibe o gráfico
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