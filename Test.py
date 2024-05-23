import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

if __name__ == "__main__":

    df = pd.read_csv('Hotel_Reviews.csv')
    df.dropna(inplace=True)

    X_train = df.loc[:4999, ['Average_Score', 'Review_Total_Negative_Word_Counts', 'Review_Total_Positive_Word_Counts']]

    clustering = DBSCAN(eps=10, min_samples=5).fit(X_train)
    DBSCAN_dataset = X_train.copy()
    DBSCAN_dataset['Cluster'] = clustering.labels_

    outliers = DBSCAN_dataset[DBSCAN_dataset['Cluster'] == -1]

    features = X_train.columns
    num_features = len(X_train.columns)
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
