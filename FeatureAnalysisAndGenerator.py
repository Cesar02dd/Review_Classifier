import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from FeatureAnalysis import FeatureAnalysis

class FeatureAnalysisAndGenerator(FeatureAnalysis):
    def __init__(self, data_loader):
        """
        Initialize the FeatureAnalysisAndGenerator object.

        Parameters:
        - data_loader: An instance of DataLoader providing access to the dataset.
        """
        super().__init__(data_loader)

    def generate_features_dataset(self):
        """
        Generate new features for both training and testing datasets.
        """
        self.generate_features(self.data_loader.data_train)
        self.generate_features(self.data_loader.data_test)

    def generate_features(self, data):
        """
        Generate new features based on the existing dataset.

        Parameters:
        - data: The dataset for which new features will be generated.
        """

        # New Features
        # Simple Combinations

        mean_words_positive = data['Review_Total_Positive_Word_Counts'].mean()
        print('Mean of Positive Words: ', mean_words_positive)
        mean_words_negative = data['Review_Total_Negative_Word_Counts'].mean()
        print('Mean of Negative Words: ', mean_words_negative)

        data['Ratio_WordCount_Between_Positive_Negative_Reviews'] = \
            (data['Review_Total_Positive_Word_Counts'] / data['Review_Total_Negative_Word_Counts'])

        data['Ratio_Between_AverageScore_TotalReviews'] = \
            (data['Review_Count_Per_Hotel_Encoded'] / data['Average_Score'])

        data['Total_Tags'] = data['Tags'].apply(self.count_tags)

        data['Total_Review_Words'] = data['Review_Total_Positive_Word_Counts'] + data['Review_Total_Negative_Word_Counts']

        data['Ratio_Total_Positive_Words'] = data['Review_Total_Positive_Word_Counts'] / data['Total_Review_Words']

        data['Ratio_Positive_Words_Mean'] = data['Review_Total_Positive_Word_Counts'] / mean_words_positive

        data['Ratio_Total_Negative_Words'] = data['Review_Total_Negative_Word_Counts'] / data['Total_Review_Words']

        data['Ratio_Negative_Words_Mean'] = data['Review_Total_Negative_Word_Counts'] / mean_words_negative

        data['Reviewer_Has_Given_Positive_Review'] = data['Review_Total_Positive_Word_Counts'].apply(
            lambda review: 0 if review == 0 else 1)

        data['Reviewer_Has_Given_Negative_Review'] = data['Review_Total_Negative_Word_Counts'].apply(
            lambda review: 0 if review == 0 else 1)

        data['Square_Ratio_WordCount'] = np.power(data['Ratio_WordCount_Between_Positive_Negative_Reviews'], 2)

        # Display the dataset after creating interactions
        print('Dataset after creating interactions: \n', data.head(20))

    def count_tags(self, tags):
        """
        Count the number of tags in a string.

        Parameters:
        - tags: A string containing tags separated by commas.

        Returns:
        - int: The number of tags.
        """
        array_tags = tags.split(',')
        return len(array_tags)

