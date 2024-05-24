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
    """
    A class responsible for performing clustering algorithms on hotel review data.

    Attributes:
        data_loader (DataLoader): An object of the DataLoader class containing the dataset.
        _data_train (pd.DataFrame): Training data containing specific columns from the data loader.
        _labels_train (pd.Series): Labels for the training data.
        _data_test (pd.DataFrame): Test data containing specific columns from the data loader.
        _labels_test (pd.Series): Labels for the test data.

    Methods:
        dbscan(): Performs DBSCAN clustering on hotel review data and visualizes clusters.
        KMeansClustering(): Performs KMeans clustering on data and visualizes clusters.
        _plot_clusters(): Internal method to plot clusters in 3D.
    """

    def __init__(self, data_loader):
        """
        Initializes the Clustering class with a DataLoader object.
        """
        self.data_loader = data_loader
        # Select specific columns for training and testing data
        self._data_train = self.data_loader.data_train[['Positive_Review', 'Negative_Review']]
        self._labels_train = self.data_loader.labels_train
        self._data_test = self.data_loader.data_test[['Positive_Review', 'Negative_Review']]
        self._labels_test = self.data_loader.labels_test

    def dbscan(self):
        """
        Performs DBSCAN clustering on hotel review data and visualizes clusters.
        """
        # Load hotel review data from a CSV file
        df = pd.read_csv('Hotel_Reviews.csv')

        # Remove rows with missing values (NA)
        df.dropna(inplace=True)

        # Select the first 5000 rows and specified columns as training set
        X_train = df.loc[:4999, ['Average_Score', 'Review_Total_Negative_Word_Counts', 'Review_Total_Positive_Word_Counts']]

        # Apply DBSCAN clustering algorithm to the training set
        clustering = DBSCAN(eps=10, min_samples=5).fit(X_train)

        # Save the clustering model to a file using pickle
        with open('Models/DBSCAN.pkl', 'wb') as dt_file:
            pickle.dump(clustering, dt_file)

        # Create a copy of the training set and add a 'Cluster' column with DBSCAN labels
        DBSCAN_dataset = X_train.copy()
        DBSCAN_dataset['Cluster'] = clustering.labels_

        # Identify outliers (points with label -1)
        outliers = DBSCAN_dataset[DBSCAN_dataset['Cluster'] == -1]

        # Extract features used for clustering
        features = X_train.columns
        num_features = len(X_train.columns)

        # Generate all possible combinations of two features
        combinations_list = list(combinations(features, 2))

        # Create a grid of subplots to visualize feature combinations
        fig, axes = plt.subplots(num_features, num_features, figsize=(6 * num_features, 6 * num_features))

        # Iterate over all features to fill the subplot grid
        for i in range(num_features):
            for j in range(num_features):
                if i != j:
                    feature_x, feature_y = features[i], features[j]
                    # Remove duplicate combinations
                    if (feature_y, feature_x) in combinations_list:
                        combinations_list.remove((feature_y, feature_x))
                    # Plot clustering data, coloring by cluster label
                    sns.scatterplot(x=feature_x, y=feature_y,
                                    data=DBSCAN_dataset[DBSCAN_dataset['Cluster'] != -1],
                                    hue='Cluster', ax=axes[i, j], palette='Set2', legend='full', s=200)
                    # Plot outliers as black points
                    axes[i, j].scatter(outliers[feature_x], outliers[feature_y], s=10, label='outliers', c="k")
                    axes[i, j].legend()
                    axes[i, j].set_title(f'{feature_x} vs {feature_y}')
                    plt.setp(axes[i, j].get_legend().get_texts(), fontsize='12')
                else:
                    # Turn off subplot for the main diagonal
                    axes[i, j].axis('off')

        # Adjust subplot layout to avoid overlap
        plt.tight_layout()
        # Show the plot
        plt.show()

    def KMeansClustering(self, data_train, data_test, n_clusters=3):
        """
        Performs KMeans clustering on data and visualizes clusters.

        Args:
            data_train (pd.DataFrame): Training data.
            data_test (pd.DataFrame): Test data.
            n_clusters (int): Number of clusters.

        Returns:
            KMeans: Trained KMeans model.
        """
        # Select only numeric columns
        numeric_data_train = data_train.select_dtypes(include=['number'])
        numeric_data_test = data_test.select_dtypes(include=['number'])

        # Initialize KMeans clustering with specified number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(numeric_data_train)

        # Predict clusters for training and test data
        train_clusters = kmeans.predict(numeric_data_train)
        test_clusters = kmeans.predict(numeric_data_test)

        # Add cluster labels to the original data as a new field 'Clustering'
        data_train['Clustering'] = train_clusters
        data_test['Clustering'] = test_clusters

        # Print cluster centers and cluster distribution for training and test data
        print(f"Cluster centers:\n{kmeans.cluster_centers_}")
        print(f"Train data cluster distribution:\n{pd.Series(train_clusters).value_counts()}")
        print(f"Test data cluster distribution:\n{pd.Series(test_clusters).value_counts()}")

        # Plot clusters
        self._plot_clusters(numeric_data_train, train_clusters)

        return kmeans

    def _plot_clusters(self, data, clusters):
        """
        Plots clusters in 3D.

        Args:
            data (pd.DataFrame): Data.
            clusters (array-like): Cluster labels.
        """
        # Reduce data to three dimensions for 3D plotting
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