from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataPreprocessing:
    """
    A class responsible for preprocessing data by normalizing numerical features.

    Attributes:
        data_loader (DataLoader): An object of the DataLoader class containing the dataset.

    Methods:
        _normalize_features(): Normalizes numerical features using StandardScaler.
    """

    def __init__(self, data_loader):
        """
        Initializes the DataPreprocessing class with a DataLoader object.

        Args:
            data_loader (DataLoader): An object of the DataLoader class containing the dataset.
        """
        self.data_loader = data_loader

        # Preprocess data
        self._normalize_features()

    def _normalize_features(self):
        """
        Normalizes numerical features using StandardScaler.
        """
        try:
            # Check if data_train and data_test are not None
            if self.data_loader.data_train is None or self.data_loader.data_test is None:
                raise ValueError("Data has not been loaded yet.")
            # Check if labels_train and labels_test are not None
            if self.data_loader.labels_train is None or self.data_loader.labels_test is None:
                raise ValueError("Labels have not been loaded yet.")

            # Select numeric columns
            numerical_columns = self.data_loader.data_train.select_dtypes(include=['number'])

            # Get the names of numeric columns
            numerical_features = list(numerical_columns.columns)

            # Normalize real numeric features using StandardScaler
            scaler = StandardScaler()
            self.data_loader.data_train[numerical_features] = scaler.fit_transform(
                self.data_loader.data_train[numerical_features])
            self.data_loader.data_test[numerical_features] = scaler.transform(
                self.data_loader.data_test[numerical_features])

            print("Features normalized successfully.")

        except ValueError as ve:
            print("Error:", ve)




