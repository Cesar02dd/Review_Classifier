import numpy as np
import pandas as pd

from FeatureAnalysis import FeatureAnalysis


class FeatureAnalysisAndGenerator(FeatureAnalysis):
    def __init__(self, data_loader):
        super().__init__(data_loader)

    def generate_features(self):
        # Choose the  column for binning
        feature_to_bin2 = 'Total_Number_of_Reviews_Reviewer_Has_Given'
        feature_to_bin3 = 'Total_Number_of_Reviews'

        # Define the number of bins (or bin edges)
        bins_expertise_level = [0, 5, 10, 15, 20, float('inf')]
        bins_total_reviews = [0, 2500, 5000, 7500, float('inf')]

        # Perform binning using pandas
        self.data_loader.data['Reviewer_Expertise_Level_bin'] = pd.cut(self.data_loader.data[feature_to_bin2],
                                                               bins=bins_expertise_level,
                                                               labels=['Nenhum', 'Baixo', 'Medio', 'Alto', 'Experto'])
        self.data_loader.data['Review_Count_Per_Hotel_bin'] = pd.cut(self.data_loader.data[feature_to_bin3], bins=bins_total_reviews,
                                                             labels=['Poucas_Avaliações', 'Algumas_Avaliações',
                                                                     'Muitas_Avaliações', 'Muitissimas_Avaliações'])

        # Display the dataset after binning
        print('Dataset after binning expertise level: \n',
              self.data_loader.data[['Total_Number_of_Reviews_Reviewer_Has_Given', 'Reviewer_Expertise_Level_bin']].head(20))
        print('Dataset after binning expertise level: \n',
              self.data_loader.data[['Total_Number_of_Reviews', 'Review_Count_Per_Hotel_bin']])

        # New Features
        # Simple Combinations

        mean_words_positive = self.data_loader.data['Review_Total_Positive_Word_Counts'].mean()
        print('Mean of Positive Words: ', mean_words_positive)
        mean_words_negative = self.data_loader.data['Review_Total_Negative_Word_Counts'].mean()
        print('Mean of Negative Words: ', mean_words_negative)

        self.data_loader.data['Ratio_WordCount_Between_Positive_Negative_Reviews'] = \
            (self.data_loader.data['Review_Total_Positive_Word_Counts'] / self.data_loader.data['Review_Total_Negative_Word_Counts'])

        #self.data_loader.data['Deviation_Between_AverageScore_ReviewerScore'] = (
        #        self.data_loader.data['Reviewer_Score'] - self.data_loader.data['Average_Score'])

        self.data_loader.data['Ratio_Between_AverageScore_TotalReviews'] = \
            (self.data_loader.data['Total_Number_of_Reviews'] / self.data_loader.data['Average_Score'])

        self.data_loader.data['Total_Tags'] = self.data_loader.data['Tags'].apply(self.count_tags)

        self.data_loader.data['Total_Review_Words'] = self.data_loader.data['Review_Total_Positive_Word_Counts'] + self.data_loader.data[
            'Review_Total_Negative_Word_Counts']

        self.data_loader.data['Ratio_Total_Positive_Words'] = self.data_loader.data['Review_Total_Positive_Word_Counts'] / \
                                                      self.data_loader.data[
                                                          'Total_Review_Words']

        self.data_loader.data['Ratio_Positive_Words_Mean'] = self.data_loader.data[
                                                         'Review_Total_Positive_Word_Counts'] / mean_words_positive

        self.data_loader.data['Ratio_Total_Negative_Words'] = self.data_loader.data['Review_Total_Negative_Word_Counts'] / \
                                                      self.data_loader.data[
                                                          'Total_Review_Words']

        self.data_loader.data['Ratio_Negative_Words_Mean'] = self.data_loader.data[
                                                         'Review_Total_Negative_Word_Counts'] / mean_words_negative

        self.data_loader.data['Reviewer_Has_Given_Positive_Review'] = self.data_loader.data['Review_Total_Positive_Word_Counts'].apply(
            lambda review: 0 if review == 0 else 1)

        self.data_loader.data['Reviewer_Has_Given_Negative_Review'] = self.data_loader.data['Review_Total_Negative_Word_Counts'].apply(
            lambda review: 0 if review == 0 else 1)

        #self.data_loader.data['Log_Reviewer_Score'] = np.log(self.data_loader.data['Reviewer_Score'])

        self.data_loader.data['Square_Ratio_WordCount'] = np.power(
            self.data_loader.data['Ratio_WordCount_Between_Positive_Negative_Reviews'], 2)

        #self.data_loader.data['Weighted_Average_Reviewer_Score'] = ((self.data_loader.data['Reviewer_Score'] *
        #                                                    self.data_loader.data[
        #                                                        'Total_Number_of_Reviews_Reviewer_Has_Given']) +
        #                                                   self.data_loader.data['Log_Reviewer_Score'])

        # Display the dataset after creating interactions
        print('Dataset after creating interactions: \n', self.data_loader.data.head(20))

    def count_tags(self, tags):
        array_tags = tags.split(',')
        return len(array_tags)
