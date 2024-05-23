import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



class Clustering:
    def __init__(self):
        pass

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
