import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from FeatureAnalysis import FeatureAnalysis


class FeatureAnalysisAndGenerator(FeatureAnalysis):
    def __init__(self, data_loader):
        super().__init__(data_loader)

    def generate_features_dataset(self):
        self.generate_features(self.data_loader.data_train)
        self.generate_features(self.data_loader.data_test)

    def generate_features(self, data):

        # New Features
        # Simple Combinations

        mean_words_positive = data['Review_Total_Positive_Word_Counts'].mean()
        print('Mean of Positive Words: ', mean_words_positive)
        mean_words_negative = data['Review_Total_Negative_Word_Counts'].mean()
        print('Mean of Negative Words: ', mean_words_negative)

        data['Ratio_WordCount_Between_Positive_Negative_Reviews'] = \
            (data['Review_Total_Positive_Word_Counts'] / data['Review_Total_Negative_Word_Counts'])

        #self.data_loader.data['Deviation_Between_AverageScore_ReviewerScore'] = (
        #        self.data_loader.data['Reviewer_Score'] - self.data_loader.data['Average_Score'])

        data['Ratio_Between_AverageScore_TotalReviews'] = \
            (data['Review_Count_Per_Hotel_Encoded'] / data['Average_Score'])

        data['Total_Tags'] = data['Tags'].apply(self.count_tags)

        data['Total_Review_Words'] = data['Review_Total_Positive_Word_Counts'] + data[
            'Review_Total_Negative_Word_Counts']

        data['Ratio_Total_Positive_Words'] = data['Review_Total_Positive_Word_Counts'] / \
                                                      data[
                                                          'Total_Review_Words']

        data['Ratio_Positive_Words_Mean'] = data[
                                                         'Review_Total_Positive_Word_Counts'] / mean_words_positive

        data['Ratio_Total_Negative_Words'] = data['Review_Total_Negative_Word_Counts'] / \
                                                      data[
                                                          'Total_Review_Words']

        data['Ratio_Negative_Words_Mean'] = data[
                                                         'Review_Total_Negative_Word_Counts'] / mean_words_negative

        data['Reviewer_Has_Given_Positive_Review'] = data['Review_Total_Positive_Word_Counts'].apply(
            lambda review: 0 if review == 0 else 1)

        data['Reviewer_Has_Given_Negative_Review'] = data['Review_Total_Negative_Word_Counts'].apply(
            lambda review: 0 if review == 0 else 1)

        #self.data_loader.data['Log_Reviewer_Score'] = np.log(self.data_loader.data['Reviewer_Score'])

        data['Square_Ratio_WordCount'] = np.power(
            data['Ratio_WordCount_Between_Positive_Negative_Reviews'], 2)

        #self.data_loader.data['Weighted_Average_Reviewer_Score'] = ((self.data_loader.data['Reviewer_Score'] *
        #                                                    self.data_loader.data[
        #                                                        'Total_Number_of_Reviews_Reviewer_Has_Given']) +
        #                                                   self.data_loader.data['Log_Reviewer_Score'])

        # Display the dataset after creating interactions
        print('Dataset after creating interactions: \n', data.head(20))

    def count_tags(self, tags):
        array_tags = tags.split(',')
        return len(array_tags)
