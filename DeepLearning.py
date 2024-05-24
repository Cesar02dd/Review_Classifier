import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

class DeepLearning:
    """
    A class for implementing deep learning models.

    Attributes:
        data_loader (DataLoader): An object of the DataLoader class containing the dataset.
        _data_train (DataFrame): Training data containing 'Positive_Review' and 'Negative_Review' columns.
        _labels_train (DataFrame): Training labels.
        _data_test (DataFrame): Test data containing 'Positive_Review' and 'Negative_Review' columns.
        _labels_test (DataFrame): Test labels.

    Methods:
        rnn(): Builds and trains a Bidirectional LSTM model for sentiment analysis.
    """

    def __init__(self, data_loader):
        """
        Initializes the DeepLearning class with a DataLoader object.

        Args:
            data_loader (DataLoader): An object of the DataLoader class containing the dataset.
        """
        self.data_loader = data_loader
        self._data_train = self.data_loader.data_train[['Positive_Review', 'Negative_Review']]
        self._labels_train = self.data_loader.labels_train
        self._data_test = self.data_loader.data_test[['Positive_Review', 'Negative_Review']]
        self._labels_test = self.data_loader.labels_test

    def rnn(self):
        """
        Builds and trains a Bidirectional LSTM model for sentiment analysis.
        """
        # Vocabulary size
        VOCAB_SIZE = 10000

        # Text vectorization for positive reviews
        encoder_positive = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
        encoder_positive.adapt(self._data_train['Positive_Review'])
        vocab_positive = np.array(encoder_positive.get_vocabulary())

        # Text vectorization for negative reviews
        encoder_negative = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
        encoder_negative.adapt(self._data_train['Negative_Review'])
        vocab_negative = np.array(encoder_negative.get_vocabulary())

        # Positive review input layer
        input_positive = tf.keras.Input(shape=(1,), dtype=tf.string)
        embedding_positive = encoder_positive(input_positive)
        embedding_positive = tf.keras.layers.Embedding(input_dim=len(encoder_positive.get_vocabulary()),
                                                       output_dim=64,
                                                       mask_zero=True)(embedding_positive)
        lstm_positive = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(embedding_positive)

        # Negative review input layer
        input_negative = tf.keras.Input(shape=(1,), dtype=tf.string)
        embedding_negative = encoder_negative(input_negative)
        embedding_negative = tf.keras.layers.Embedding(input_dim=len(encoder_negative.get_vocabulary()),
                                                       output_dim=64,
                                                       mask_zero=True)(embedding_negative)
        lstm_negative = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(embedding_negative)

        # Concatenate positive and negative review representations
        concatenated = tf.keras.layers.Concatenate()([lstm_positive, lstm_negative])

        # Dense layers
        dense1 = tf.keras.layers.Dense(64, activation='relu')(concatenated)
        output = tf.keras.layers.Dense(1)(dense1)

        # Model
        model = tf.keras.Model(inputs=[input_positive, input_negative], outputs=output)

        # Compile the model
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(1e-4),
                      metrics=['accuracy'])

        # Train the model
        history = model.fit([self._data_train['Positive_Review'], self._data_train['Negative_Review']],
                            self._labels_train,
                            epochs=10,
                            validation_split=0.3,
                            shuffle=True)

        # Evaluate the model
        test_loss, test_acc = model.evaluate([self._data_test['Positive_Review'], self._data_test['Negative_Review']],
                                             self._labels_test)
        print('Test Loss:', test_loss)
        print('Test Accuracy:', test_acc)

        # Plot training history
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label="Accuracy")
        plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label="Loss")
        plt.plot(history.history['val_loss'], label="Validation Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

