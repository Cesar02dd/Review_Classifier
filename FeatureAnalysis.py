import numpy as np
import umap
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import LocallyLinearEmbedding, TSNE


class FeatureAnalysis:
    def __init__(self, data_loader):
        self.data_loader = data_loader

        self._data = self.data_loader.data.select_dtypes(include=['number'])
        self._targets = self.data_loader.labels['Reviewer_Score_bin_encoded']

    def perform_feature_analysis(self):
        # Compute and plot PCA projection
        print('PCA')
        self.plot_projection(self.compute_pca(), 'PCA Projection')
        # Compute and plot LDA projection
        print('LDA')
        self.plot_projection(self.compute_lda(), 'LDA Projection')
        # Compute and plot t-SNE projection
        print('t-SNE')
        self.plot_projection(self.compute_tsne(), 't-SNE Projection')
        # Compute and plot UMAP projection
        print('UMAP')
        self.plot_projection(self.compute_umap(), 'UMAP Projection')
        # Compute and plot LLE projection
        # print('LLE')
        # self.plot_projection(self.compute_lle(), 'LLE Projection')

    def compute_pca(self, n_components=2):
        """
        Compute Principal Component Analysis (PCA) on the dataset.

        Parameters:
        - n_components: The number of components to keep.

        Returns:
        - pca_projection: The projected data using PCA.
        """
        return PCA(n_components=n_components).fit_transform(self._data)

    def compute_lda(self, n_components=2):
        """
        Perform Linear Discriminant Analysis (LDA) on the input data.

        Parameters:
        - n_components: The number of components to keep

        Returns:
            array-like: The reduced-dimensional representation of the data using LDA.
        """
        return LinearDiscriminantAnalysis(n_components=n_components).fit_transform(self._data, self._targets)

    def compute_tsne(self, n_components=2, perplexity=3):

        """
        Compute t-Distributed Stochastic Neighbor Embedding (t-SNE) on the dataset.

        Parameters:
        - n_components: The number of components to embed the data into.
        - perplexity: The perplexity parameter for t-SNE.

        Returns:
        - tsne_projection: The projected data using t-SNE.
        """
        return TSNE(n_components=n_components, perplexity=perplexity).fit_transform(self._data)

    def compute_umap(self, n_components=2, n_neighbors=8, min_dist=0.5, metric='euclidean'):
        """
        Compute Uniform Manifold Approximation and Projection (UMAP) on the dataset.

        Parameters:
        - n_components: The number of components to embed the data into.
        - n_neighbors: The number of neighbors to consider for each point.
        - min_dist: The minimum distance between embedded points.
        - metric: The distance metric to use.

        Returns:
        - umap_projection: The projected data using UMAP.
        """
        return umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
                         metric=metric).fit_transform(self._data)

    def compute_lle(self, n_components=2, n_neighbors=20):
        """
        Compute Locally Linear Embedding (LLE) on the dataset.

        Parameters:
        - n_components: The number of components to embed the data into.
        - n_neighbors: The number of neighbors to consider for each point.

        Returns:
        - lle_projection: The projected data using LLE.
        """
        return LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components).fit_transform(self._data)

    def plot_projection(self, projection, title):
        """
        Plot the 2D projection of the dataset.

        Parameters:
        - projection: The projected data.
        - title: The title of the plot.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(projection[:, 0], projection[:, 1], c=self._targets, alpha=0.5)
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        plt.show()

    def relevant_feature_identification(self, num_features=10):
        """
        Perform feature relevant feature identification using mutual information between each feature and the target
        variable. Mutual information measures the amount of information obtained about one random variable through
        another random variable. It quantifies the amount of uncertainty reduced for one variable given the knowledge
        of another variable. In feature selection, mutual information helps identify the relevance of features with
        respect to the target variable.

        Parameters:
            num_features (int): Number of features to select.

        Returns:
            selected_features (list): List of selected feature names.
        """
        try:
            # Check if data_train is not None
            if self.data_loader.data_train is None or self.data_loader.labels_train is None:
                raise ValueError("Training data or labels have not been loaded yet.")

            # Perform feature selection using mutual information
            mutual_info = mutual_info_classif(self.data_loader.data_train, self.data_loader.labels_train)

            selected_features_indices = np.argsort(mutual_info)[::-1][:num_features]
            selected_features = self.data_loader.data_train.columns[selected_features_indices]

            print(f"{num_features} relevant features identified.")
            print(selected_features.tolist())
            return selected_features.tolist()

        except ValueError as ve:
            print("Error:", ve)

