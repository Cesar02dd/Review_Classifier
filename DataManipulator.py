import pandas as pd
from sklearn.preprocessing import LabelEncoder

from DataLoader import DataLoader


class DataManipulator(DataLoader):
    def __init__(self, filepath, test_size=0.2, state=None):
        super().__init__(filepath, test_size, state)

        # Manipulate data
        self.drop_columns()
        self._encode_score()

    def drop_columns(self):
        try:
            self.data = self.data.drop(columns=['Id_Hotel_Rating', 'Additional_Number_of_Scoring'])
            self.data_test = self.data_test.drop(columns=['Id_Hotel_Rating', 'Additional_Number_of_Scoring'])
            self.data_train = self.data_train.drop(columns=['Id_Hotel_Rating', 'Additional_Number_of_Scoring'])

        except Exception as e:
            print("Error:", e)

    def _encode_score(self):
        try:
            # Select Reviewer Score column
            feature_to_bin = 'Reviewer_Score'

            # Define the number of bins (or bin edges)
            bins_reviewer_score = [0, 4, 7, 10]

            # Perform binning using pandas
            self.labels['Reviewer_Score_bin'] = pd.cut(self.labels[feature_to_bin], bins=bins_reviewer_score,
                                                       labels=['Mau', 'Neutral', 'Bom'])
            self.labels_train['Reviewer_Score_bin'] = pd.cut(self.labels_train[feature_to_bin],
                                                             bins=bins_reviewer_score,
                                                             labels=['Mau', 'Neutral', 'Bom'])
            self.labels_test['Reviewer_Score_bin'] = pd.cut(self.labels_test[feature_to_bin],
                                                            bins=bins_reviewer_score,
                                                            labels=['Mau', 'Neutral', 'Bom'])

            # Create an instance of LabelEncoder
            label_encoder = LabelEncoder()

            # Encode the 'reviewer_score_bin' column
            self.labels['Reviewer_Score_bin_encoded'] = label_encoder.fit_transform(
                self.labels['Reviewer_Score_bin'])

            self.labels_train['Reviewer_Score_bin_encoded'] = label_encoder.fit_transform(
                self.labels_train['Reviewer_Score_bin'])

            self.labels_test['Reviewer_Score_bin_encoded'] = label_encoder.fit_transform(
                self.labels_test['Reviewer_Score_bin'])

            # Display the dataset after binning
            print('Dataset after binning score reviewer: \n', self.labels['Reviewer_Score', 'Reviewer_Score_bin', 'Reviewer_Score_bin_encoded'])

        except Exception as e:
            print("Error:", e)
