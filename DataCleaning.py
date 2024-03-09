import numpy as np


class DataCleaning:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def remove_duplicates(self):
        try:
            # Check if data and labels are not None
            if self.data_loader.data_train is None:
                raise ValueError("Data has not been loaded yet.")
            if self.data_loader.labels_train is None:
                raise ValueError("Labels have not been loaded yet.")

            # Remove duplicate rows from training data (do not apply to test data)
            self.data_loader.data_train.drop_duplicates(inplace=True)
            self.data_loader.labels_train = self.data_loader.labels_train[self.data_loader.data_train.index]

            print("Duplicate rows removed from training data.")

        except ValueError as ve:
            print("Error:", ve)

    def handle_missing_values(self, strategy='drop'):
        try:
            # Check if data is not None
            if self.data_loader.data_train is None or self.data_loader.data_test is None:
                raise ValueError("Data has not been loaded yet.")

            # Check if there are missing values
            if self.data_loader.data_train.isnull().sum().sum() == 0 and self.data_loader.data_test.isnull().sum().sum() == 0:
                print("No missing values found in the data.")
                return

            # Handle missing values based on the specified strategy
            if strategy == 'mean':
                self.data_loader.data_train.fillna(self.data_loader.data_train.mean(), inplace=True)
                self.data_loader.data_test.fillna(self.data_loader.data_test.mean(), inplace=True)
            elif strategy == 'median':
                self.data_loader.data_train.fillna(self.data_loader.data_train.median(), inplace=True)
                self.data_loader.data_test.fillna(self.data_loader.data_test.median(), inplace=True)
            elif strategy == 'most_frequent':
                self.data_loader.data_train.fillna(self.data_loader.data_train.mode().iloc[0], inplace=True)
                self.data_loader.data_test.fillna(self.data_loader.data_test.mode().iloc[0], inplace=True)
            elif strategy == 'fill_nan':
                self.data_loader.data_train.fillna(strategy, inplace=True)
                self.data_loader.data_test.fillna(strategy, inplace=True)
            elif strategy == 'drop':
                self.data_loader.data_train = self.data_loader.data_train.dropna(axis=0)
                self.data_loader.labels_train = self.data_loader.labels_train[self.data_loader.data_train.index]
                self.data_loader.data_test = self.data_loader.data_test.dropna(axis=0)
                self.data_loader.labels_test = self.data_loader.labels_test[self.data_loader.data_test.index]

            else:
                raise ValueError("Invalid strategy.")
            print("Missing values handled using strategy:", strategy)

        except ValueError as ve:
            print("Error:", ve)

    def _detect_outliers(self):
        try:
            # Check if test data is not None
            if self.data_loader.data_train is None:
                raise ValueError("Data has not been loaded yet.")

            # Dealing with outliers for all numerical columns
            outliers = None

            print("\nDetecting outliers:")
            for feature in self.data_loader.data_train.select_dtypes(include=[np.number]).columns:
                # Calculate the IQR (InterQuartile Range)
                Q1 = self.data_loader.data_train[feature].quantile(0.25)
                Q3 = self.data_loader.data_train[feature].quantile(0.75)
                IQR = Q3 - Q1

                # Define the lower and upper bounds to identify outliers
                upper_bound = Q3 + 1.5 * IQR
                lower_bound = Q1 - 1.5 * IQR

                outliers = self.data_loader.data_train[
                    (self.data_loader.data_train[feature] < lower_bound) | (self.data_loader.data_train[feature] > upper_bound)]
                # Each row represents a data point where the values for the features are listed.
                # The first row presents the indices of the DataFrame where the outliers were detected.
                print(f"Outliers in '{feature}':\n{outliers}" if not outliers.empty else f"No outliers in '{feature}'.")

            # Display dataset after handling outliers for all numerical columns
            print(self.data_loader.data_train)

            return outliers

        except Exception as e:
            print("Error:", e)

    def remove_outliers(self):
        try:
            # Check if data_loader.data_train is not None
            if self.data_loader.data_train is None:
                raise ValueError("Data has not been loaded yet.")

            # Detect outliers
            outliers = self._detect_outliers()

            # Remove outliers from the dataset
            self.data_loader.data_train = self.data_loader.data_train.drop(outliers.index)
            self.data_loader.labels_train = self.data_loader.labels_train[self.data_loader.data_train.index]

            print("Outliers removed from the dataset.")

        except Exception as e:
            print("Error:", e)
