import warnings

import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:

    def __init__(self, filepath, test_size=0.2, state=None):

        self.filepath = filepath
        self.test_size = test_size
        self.state = state

        self.data = None
        self.labels = None
        self.data_train = None
        self.labels_train = None
        self.data_test = None
        self.labels_test = None

        # Load Data
        self._load_data()

    def _load_data(self):
        try:
            # Ignore
            warnings.simplefilter(action='ignore', category=FutureWarning)

            # Load the Dataset
            df = pd.read_csv(self.filepath, delimiter=',')
            self.dataframe = df

            # Split the data into features and labels
            self.data = df.drop(columns=['Reviewer_Score'])
            #self.target = df[['Reviewer_Score']]
            self.labels = df[['Reviewer_Score']].copy()

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=self.test_size,
                                                                random_state=self.state)

            # Assign the data and labels to attributes
            self.data_train = X_train
            self.labels_train = y_train
            self.data_test = X_test
            self.labels_test = y_test

            print("Data loaded successfully.")

        except FileNotFoundError as e:
            print("File not found. Please check the file path.")


