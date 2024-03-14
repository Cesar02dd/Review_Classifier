import pandas as pd
from sklearn.preprocessing import LabelEncoder

from DataLoader import DataLoader


class DataManipulator(DataLoader):
    def __init__(self, filepath, test_size=0.2, state=None):
        super().__init__(filepath, test_size, state)

        # Manipulate data
        self.drop_columns()

    def drop_columns(self):
        try:
            # self.dataframe = self.dataframe.drop(columns=['Id_Hotel_Rating', 'Additional_Number_of_Scoring'])
            self.data = self.data.drop(columns=['Id_Hotel_Rating', 'Additional_Number_of_Scoring'])
            self.data_test = self.data_test.drop(columns=['Id_Hotel_Rating', 'Additional_Number_of_Scoring'])
            self.data_train = self.data_train.drop(columns=['Id_Hotel_Rating', 'Additional_Number_of_Scoring'])

        except Exception as e:
            print("Error:", e)


