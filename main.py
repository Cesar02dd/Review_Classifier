import pickle

from Clustering import Clustering
from DataCleaning import DataCleaning
from DataManipulator import DataManipulator
from DataPreprocessing import DataPreprocessing
from DataVisualization import DataVisualization
from DeepLearning import DeepLearning
from EDA import EDA
from FeatureAnalysisAndGenerator import FeatureAnalysisAndGenerator
from HypothesisTesting import HypothesisTesting
from KNN import KNN
from PreVisualization import PreVisualization
from SupervisedLearning import SupervisedLearning
from EnsembleModel import EnsembleModel

if __name__ == "__main__":

    data_loader = DataManipulator("Hotel_Reviews.csv")
    data_preprocessing = DataPreprocessing(data_loader)

    # First look at the data
    pre_visualization = PreVisualization(data_loader)

    # Access the data and labels attributes
    print("\n\nBefore data preprocessing")
    print("Normal data shape:", data_loader.data.shape)
    print("Normal labels shape:", data_loader.labels.shape)
    print("Training data shape:", data_loader.data_train.shape)
    print("Training labels shape:", data_loader.labels_train.shape)
    print("Testing data shape:", data_loader.data_test.shape)
    print("Testing labels shape:", data_loader.labels_test.shape)

    data_cleaner = DataCleaning(data_loader)
    data_cleaner.remove_duplicates()
    data_cleaner.handle_missing_values()
    data_cleaner.remove_outliers()

    print("\n\nAfter data preprocessing")
    print("Training data shape:", data_loader.data_train.shape)
    print("Training labels shape:", data_loader.labels_train.shape)
    print("Testing data shape:", data_loader.data_test.shape)
    print("Testing labels shape:", data_loader.labels_test.shape)

    # Serialize data_loader object to save a copy of the cleaned data
    with open('data_loader.pkl', 'wb') as f:
        pickle.dump(data_loader, f)

    # Deserialize data_loader object to restore the cleaned data
    with open('data_loader.pkl', 'rb') as f:
        data_loader_loaded = pickle.load(f)
    print("\n\nDeserialized data")
    print("Training data shape:", data_loader_loaded.data_train.shape)
    print("Training labels shape:", data_loader_loaded.labels_train.shape)
    print("Testing data shape:", data_loader_loaded.data_test.shape)
    print("Testing labels shape:", data_loader_loaded.labels_test.shape)

    # EDA
    eda = EDA(data_loader)
    #eda.perform_eda()

    data_visualization = DataVisualization(data_loader)
    #data_visualization.perform_visualization()

    # Feature Analysis
    feature_analysis = FeatureAnalysisAndGenerator(data_loader)

    # Perform Analysis
    #feature_analysis.perform_feature_analysis()

    # Generate new features
    feature_analysis.generate_features_dataset()

    # Perform relevant feature identification
    columns = data_loader.data.select_dtypes(include=['number'])
    relevant_features = feature_analysis.relevant_feature_identification(len(columns))
    relevant_features_mrmr = feature_analysis.select_features_mrmr(len(columns))

    # Modified data sample visualization
    #data_visualization.plot_boxplot()

    # Serialize data_loader object to save a copy of the cleaned data with new features
    with open('data_loader_with_new_features.pkl', 'wb') as f:
        pickle.dump(data_loader, f)

    # Deserialize data_loader object to restore the cleaned data
    with open('data_loader_with_new_features.pkl', 'rb') as f:
        data_loader_loaded = pickle.load(f)
    print("Training data shape:", data_loader_loaded.data_train.shape)
    print("Training labels shape:", data_loader_loaded.labels_train.shape)
    print("Testing data shape:", data_loader_loaded.data_test.shape)
    print("Testing labels shape:", data_loader_loaded.labels_test.shape)

    # Hypothesis Testing
    hypothesis_tester = HypothesisTesting(data_loader)
    #hypothesis_tester.anova_results()
    #hypothesis_tester.t_test_results()

    knn = KNN(data_loader)
    print("Performing kNN")
    #knn.knn_compare()

    supervised_learning = SupervisedLearning(data_loader)
    print("Performing Supervised Learning Algorithms")
    supervised_learning.DecisionsTrees()
    supervised_learning.MLPClassifier()
    supervised_learning.XGBClassifier()
    supervised_learning.MulticlassClassifier()

    print("\n\nDeep Learning Model")
    deep_learning = DeepLearning(data_loader)
    print("\n\nRNN: ")
    deep_learning.rnn()

    # Ensemble Model
    ensemble_model = EnsembleModel(data_loader)
    print("\n\nPerforming Ensemble Learning Algorithms")

    print("\nVoting Classifier:")
    ensemble_model.VotingClassifier()

    print("\nGradient Boosting Classifier:")
    ensemble_model.GradientBoostingClassifier()

    print("\nRandom Forest Classifier:")
    ensemble_model.RandomForestClassifier()

    # KMeans Clustering
    preprocessed_data_train = data_loader.data_train
    preprocessed_data_test = data_loader.data_test

    # Clustering
    clustering_model = Clustering(data_loader)
    kmeans_model = clustering_model.KMeansClustering(preprocessed_data_train, preprocessed_data_test, n_clusters=2)
    print(f"Cluster centers:\n{kmeans_model.cluster_centers_}")

    clustering_model.dbscan()

