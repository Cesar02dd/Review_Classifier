import pickle

from DataCleaning import DataCleaning
from DataManipulator import DataManipulator
from DataPreprocessing import DataPreprocessing
from DataVisualization import DataVisualization
from EDA import EDA
from FeatureAnalysisAndGenerator import FeatureAnalysisAndGenerator
from HypothesisTesting import HypothesisTesting
from PreVisualization import PreVisualization

if __name__ == "__main__":

    data_loader = DataManipulator("Hotel_Reviews.csv")
    data_preprocessing = DataPreprocessing(data_loader)

    # First look at the data

    pre_visualization = PreVisualization(data_loader)

    pre_visualization.perform_pre_visualization()

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
    print("Normal data shape:", data_loader.data.shape)
    print("Normal labels shape:", data_loader.labels.shape)
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
    print("Training data shape:", data_loader_loaded.data.shape)
    print("Training labels shape:", data_loader_loaded.labels.shape)
    print("Training data shape:", data_loader_loaded.data_train.shape)
    print("Training labels shape:", data_loader_loaded.labels_train.shape)
    print("Testing data shape:", data_loader_loaded.data_test.shape)
    print("Testing labels shape:", data_loader_loaded.labels_test.shape)

    # EDA

    eda = EDA(data_loader)
    eda.perform_eda()

    data_visualization = DataVisualization(data_loader)
    data_visualization.perform_visualization()

    # Feature Analysis

    feature_analysis = FeatureAnalysisAndGenerator(data_loader)

    # Perform Analysis
    #feature_analysis.perform_feature_analysis() !!!!!!!!

    # Generate new features
    feature_analysis.generate_features()

    # Perform relevant feature identification
    relevant_features = feature_analysis.relevant_feature_identification(len(data_loader.data_train.columns))

    # Modified data sample visualization
    data_visualization.plot_boxplot()

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

    hypothesis_tester = HypothesisTesting(data_loader)
    hypothesis_tester.anova_results()
    hypothesis_tester.kruskal_wallis_results()
    hypothesis_tester.t_test_results()
