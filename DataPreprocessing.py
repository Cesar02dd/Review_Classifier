from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataPreprocessing:
    def __init__(self, data_loader):
        self.data_loader = data_loader

        # Preprocess data
        self._normalize_features()

    def _normalize_features(self):
        try:
            # Check if data_train and data_test are not None
            if self.data_loader.data_train is None or self.data_loader.data_test is None:
                raise ValueError("Data has not been loaded yet.")
            # Check if labels_train and labels_test are not None
            if self.data_loader.labels_train is None or self.data_loader.labels_test is None:
                raise ValueError("Labels have not been loaded yet.")

            # Select numeric columns
            numerical_columns = self.data_loader.data_train.select_dtypes(include=['number'])

            print("Numeric columns: " + numerical_columns.columns)

            # Select numeric features
            numerical_features = list(numerical_columns.columns)

            # Normalize real numeric features using StandardScaler
            scaler = StandardScaler()
            #self.data_loader.data[numerical_features] = scaler.fit_transform(
                #self.data_loader.data[numerical_features])
            self.data_loader.data_train[numerical_features] = scaler.fit_transform(
                self.data_loader.data_train[numerical_features])
            self.data_loader.data_test[numerical_features] = scaler.transform(
                self.data_loader.data_test[numerical_features])

            # # Identify encoded features
            # encoded_features = self.data_loader.data.columns[numerical_features]
            #
            # # Normalize encoded features using MinMaxScaler
            # scaler = MinMaxScaler()
            # self.data_loader.data[encoded_features] = scaler.fit_transform(
            #     self.data_loader.data[encoded_features])
            # self.data_loader.data_train[encoded_features] = scaler.fit_transform(
            #     self.data_loader.data_train[encoded_features])
            # self.data_loader.data_test[encoded_features] = scaler.transform(
            #     self.data_loader.data_test[encoded_features])

            print("Features normalized successfully.")

        except ValueError as ve:
            print("Error:", ve)



