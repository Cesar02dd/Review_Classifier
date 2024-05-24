import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class DataLoader:
    """
    A class responsible for loading and preprocessing data.

    Attributes:
        filepath (str): The file path of the dataset.
        test_size (float): The proportion of the dataset to include in the test split.
        state (int): Controls the shuffling applied to the data before applying the split.

    Methods:
        _load_data(): Loads the dataset, splits it into features and labels, and splits it into training and testing sets.
        _encode_score(): Bins the Reviewer_Score column, encodes it, and plots the distribution.
    """

    def __init__(self, filepath, test_size=0.2, state=None):
        """
        Initializes the DataLoader class.

        Args:
            filepath (str): The file path of the dataset.
            test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
            state (int): Controls the shuffling applied to the data before applying the split.
        """
        self.filepath = filepath
        self.test_size = test_size
        self.state = state

        self.dataframe = None
        self.data = None
        self.labels = None
        self.data_train = None
        self.labels_train = None
        self.data_test = None
        self.labels_test = None

        # Load Data
        self._load_data()

    def _load_data(self):
        """
        Loads the dataset, splits it into features and labels, and splits it into training and testing sets.
        """
        try:
            # Ignore FutureWarning
            warnings.simplefilter(action='ignore', category=FutureWarning)

            # Load the Dataset
            df = pd.read_csv(self.filepath)
            self.dataframe = df

            # Split the data into features and labels
            self.data = df.drop(columns=['Reviewer_Score'])
            self.labels = df[['Reviewer_Score']].copy()

            self._encode_score()

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

    def _encode_score(self):
        """
        Bins the Reviewer_Score column, encodes it, and plots the distribution.
        """
        try:
            # Binning Reviewer Score column
            feature_to_bin = 'Reviewer_Score'
            bins_reviewer_score = [0, 8.5, 10]

            self.labels['Reviewer_Score_bin'] = pd.cut(self.labels[feature_to_bin], bins=bins_reviewer_score,
                                                        labels=['Mau', 'Bom'])

            label_encoder = LabelEncoder()
            self.labels['Reviewer_Score_bin_encoded'] = label_encoder.fit_transform(
                self.labels['Reviewer_Score_bin'])

            # Plot bar graph
            plt.figure(figsize=(8, 6))
            counts = self.labels['Reviewer_Score_bin_encoded'].value_counts().sort_index()
            label_map = {0: 'Bom', 1: 'Mau'}
            index_labels = [label_map[idx] for idx in counts.index]
            counts.plot(kind='bar', color='skyblue')
            plt.title('Distribuição das Qualificações Y1')
            plt.xlabel('Qualificação')
            plt.ylabel('Número de Instâncias')
            plt.xticks(range(len(index_labels)), index_labels, rotation=0)
            plt.show()

            # Display the dataset after binning
            print('Dataset after binning score reviewer: \n',
                  self.labels[['Reviewer_Score', 'Reviewer_Score_bin', 'Reviewer_Score_bin_encoded']])

            self.labels = self.labels['Reviewer_Score_bin_encoded']

            # Binning Total_Number_of_Reviews_Reviewer_Has_Given and Total_Number_of_Reviews
            feature_to_bin2 = 'Total_Number_of_Reviews_Reviewer_Has_Given'
            feature_to_bin3 = 'Total_Number_of_Reviews'

            bins_expertise_level = [0, 5, 10, 15, 20, float('inf')]
            bins_total_reviews = [0, 2500, 5000, 7500, float('inf')]

            reviewer_expertise_level_bin = pd.cut(self.data[feature_to_bin2],
                                                  bins=bins_expertise_level,
                                                  labels=['Nenhum', 'Baixo', 'Medio', 'Alto', 'Experto'])
            review_count_per_hotel_bin = pd.cut(self.data[feature_to_bin3], bins=bins_total_reviews,
                                                labels=['Poucas_Avaliações', 'Algumas_Avaliações',
                                                        'Muitas_Avaliações', 'Muitissimas_Avaliações'])

            label_encoder = LabelEncoder()
            label_encoder2 = LabelEncoder()

            self.data['Reviewer_Expertise_Level_Encoded'] = label_encoder.fit_transform(
                reviewer_expertise_level_bin)

            self.data['Review_Count_Per_Hotel_Encoded'] = label_encoder2.fit_transform(
                review_count_per_hotel_bin)

            # Display the dataset after binning
            print('Dataset after binning expertise level: \n',
                  self.data[
                      ['Total_Number_of_Reviews_Reviewer_Has_Given', 'Reviewer_Expertise_Level_Encoded']].head(20))
            print('Dataset after binning expertise level: \n',
                  self.data[['Total_Number_of_Reviews', 'Review_Count_Per_Hotel_Encoded']])

            self.data = self.data.drop(columns=['Total_Number_of_Reviews_Reviewer_Has_Given', 'Total_Number_of_Reviews'])

        except Exception as e:
            print("Error:", e)
