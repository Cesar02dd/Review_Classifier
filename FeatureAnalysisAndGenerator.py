import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from FeatureAnalysis import FeatureAnalysis


class FeatureAnalysisAndGenerator(FeatureAnalysis):
    def __init__(self, data_loader):
        super().__init__(data_loader)

    def generate_features(self):

        # New Features
        # Simple Combinations

        mean_words_positive = self.data_loader.data_train['Review_Total_Positive_Word_Counts'].mean()
        print('Mean of Positive Words: ', mean_words_positive)
        mean_words_negative = self.data_loader.data_train['Review_Total_Negative_Word_Counts'].mean()
        print('Mean of Negative Words: ', mean_words_negative)

        self.data_loader.data_train['Ratio_WordCount_Between_Positive_Negative_Reviews'] = \
            (self.data_loader.data_train['Review_Total_Positive_Word_Counts'] / self.data_loader.data_train['Review_Total_Negative_Word_Counts'])

        #self.data_loader.data['Deviation_Between_AverageScore_ReviewerScore'] = (
        #        self.data_loader.data['Reviewer_Score'] - self.data_loader.data['Average_Score'])

        self.data_loader.data_train['Ratio_Between_AverageScore_TotalReviews'] = \
            (self.data_loader.data_train['Review_Count_Per_Hotel_Encoded'] / self.data_loader.data_train['Average_Score'])

        self.data_loader.data_train['Total_Tags'] = self.data_loader.data_train['Tags'].apply(self.count_tags)

        self.data_loader.data_train['Total_Review_Words'] = self.data_loader.data_train['Review_Total_Positive_Word_Counts'] + self.data_loader.data_train[
            'Review_Total_Negative_Word_Counts']

        self.data_loader.data_train['Ratio_Total_Positive_Words'] = self.data_loader.data_train['Review_Total_Positive_Word_Counts'] / \
                                                      self.data_loader.data_train[
                                                          'Total_Review_Words']

        self.data_loader.data_train['Ratio_Positive_Words_Mean'] = self.data_loader.data_train[
                                                         'Review_Total_Positive_Word_Counts'] / mean_words_positive

        self.data_loader.data_train['Ratio_Total_Negative_Words'] = self.data_loader.data_train['Review_Total_Negative_Word_Counts'] / \
                                                      self.data_loader.data_train[
                                                          'Total_Review_Words']

        self.data_loader.data_train['Ratio_Negative_Words_Mean'] = self.data_loader.data_train[
                                                         'Review_Total_Negative_Word_Counts'] / mean_words_negative

        self.data_loader.data_train['Reviewer_Has_Given_Positive_Review'] = self.data_loader.data_train['Review_Total_Positive_Word_Counts'].apply(
            lambda review: 0 if review == 0 else 1)

        self.data_loader.data_train['Reviewer_Has_Given_Negative_Review'] = self.data_loader.data_train['Review_Total_Negative_Word_Counts'].apply(
            lambda review: 0 if review == 0 else 1)

        #self.data_loader.data['Log_Reviewer_Score'] = np.log(self.data_loader.data['Reviewer_Score'])

        self.data_loader.data_train['Square_Ratio_WordCount'] = np.power(
            self.data_loader.data_train['Ratio_WordCount_Between_Positive_Negative_Reviews'], 2)

        #self.data_loader.data['Weighted_Average_Reviewer_Score'] = ((self.data_loader.data['Reviewer_Score'] *
        #                                                    self.data_loader.data[
        #                                                        'Total_Number_of_Reviews_Reviewer_Has_Given']) +
        #                                                   self.data_loader.data['Log_Reviewer_Score'])

        # Display the dataset after creating interactions
        print('Dataset after creating interactions: \n', self.data_loader.data_train.head(20))

    def count_tags(self, tags):
        array_tags = tags.split(',')
        return len(array_tags)
