import numpy as np
import umap
from mrmr import mrmr_classif
from matplotlib import pyplot as plt, cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import LocallyLinearEmbedding, TSNE


class FeatureAnalysis:
    def __init__(self, data_loader):
        self.data_loader = data_loader

        self._data = self.data_loader.data_train.select_dtypes(include=['number'])
        self._targets = self.data_loader.labels_train

    def perform_feature_analysis(self):
        # Compute and plot PCA projection
        print('PCA')
        self.plot_projection(self.compute_pca(), 'PCA Projection')
        # Compute and plot LDA projection
        print('LDA')
        self.plot_projection(self.compute_lda(), 'LDA Projection')
        # Compute and plot t-SNE projection
        print('t-SNE')
        #self.plot_projection(self.compute_tsne(), 't-SNE Projection')
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
        return PCA(n_components=n_components).fit_transform(self._data), self._data

    def compute_lda(self, n_components=2):
        """
        Perform Linear Discriminant Analysis (LDA) on the input data.

        Parameters:
        - n_components: The number of components to keep

        Returns:
            array-like: The reduced-dimensional representation of the data using LDA.
        """
        return LinearDiscriminantAnalysis(n_components=n_components).fit_transform(self._data, self._targets), self._data

    def compute_tsne(self, n_components=2, perplexity=50, learning_rate=10, n_iter=3000):

        """
        Compute t-Distributed Stochastic Neighbor Embedding (t-SNE) on the dataset.

        Parameters:
        - n_components: The number of components to embed the data into.
        - perplexity: The perplexity parameter for t-SNE.

        Returns:
        - tsne_projection: The projected data using t-SNE.
        """
        test = self._data.sample(n=10000, random_state=42)
        return TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter).fit_transform(test), test

    def compute_umap(self, n_components=2, n_neighbors=10, min_dist=0.5, metric='euclidean'):
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
        #test = self._data.sample(n=10000, random_state=42)
        test = self._data
        return umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
                         metric=metric).fit_transform(test), test

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
        plt.scatter(projection[0][:, 0], projection[0][:, 1], c=self._targets.loc[projection[1].index], alpha=0.5)

        classes = sorted(self._targets.unique())
        total_classes = len(classes)

        norm = Normalize(vmin=0, vmax=total_classes - 1)
        sm = cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])

        legend_colors = [sm.to_rgba(i) for i in range(total_classes)]

        legend_elements = [Line2D([0], [0], marker='o', color='w', label=class_,
                                  markerfacecolor=legend_colors[i])
                           for i, class_ in enumerate(classes)]

        legend1 = plt.legend(handles=legend_elements, title="Classes")

        plt.gca().add_artist(legend1)

        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        plt.savefig(title + ".png", dpi=300, bbox_inches='tight')
        plt.show()

    def relevant_feature_identification(self, num_features=7):
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

            data = self.data_loader.data_train.select_dtypes(include=['number'])

            # Perform feature selection using mutual information
            mutual_info = mutual_info_classif(data, self.data_loader.labels_train)

            selected_features_indices = np.argsort(mutual_info)[::-1][:num_features]
            selected_features = data.columns[selected_features_indices]

            print(f"{num_features} relevant features identified.")
            print(selected_features.tolist())
            return selected_features.tolist()

        except ValueError as ve:
            print("Error:", ve)

    def select_features_mrmr(self, k=5):
        """
        Select features using mRMR (minimum Redundancy Maximum Relevance).

        Parameters:
        - k (int): The number of features to select. Default is 5.

        Returns:
        - List: The selected features as a list.
        """

        data = self.data_loader.data_train.select_dtypes(include=['number'])
        # Return the selected features
        selected_features = mrmr_classif(X=data, y=self.data_loader.labels_train, K=k)

        print("Selected features (mRMR):", selected_features)

        return selected_features

