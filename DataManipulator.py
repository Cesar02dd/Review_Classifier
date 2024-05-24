import pandas as pd
from sklearn.preprocessing import LabelEncoder

from DataLoader import DataLoader


class DataManipulator(DataLoader):
    """
    A class responsible for manipulating data by dropping specific columns.

    Attributes:
        filepath (str): The file path of the dataset.
        test_size (float): The proportion of the dataset to include in the test split.
        state (int): Controls the shuffling applied to the data before applying the split.

    Methods:
        drop_columns(): Drops specific columns from the dataset.
    """

    def __init__(self, filepath, test_size=0.2, state=None):
        """
        Initializes the DataManipulator class.

        Args:
            filepath (str): The file path of the dataset.
            test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
            state (int): Controls the shuffling applied to the data before applying the split.
        """
        super().__init__(filepath, test_size, state)

        # Manipulate data
        self.drop_columns()

    def drop_columns(self):
        """
        Drops specific columns from the dataset.
        """
        try:
            # Drop specified columns from the dataset
            self.data = self.data.drop(columns=['Id_Hotel_Rating', 'Additional_Number_of_Scoring'])
            self.data_test = self.data_test.drop(columns=['Id_Hotel_Rating', 'Additional_Number_of_Scoring'])
            self.data_train = self.data_train.drop(columns=['Id_Hotel_Rating', 'Additional_Number_of_Scoring'])

        except Exception as e:
            print("Error:", e)

