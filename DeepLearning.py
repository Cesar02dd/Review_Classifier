import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models


class DeepLearning:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._data_train = self.data_loader.data_train[['Positive_Review', 'Negative_Review']]
        self._labels_train = self.data_loader.labels_train
        self._data_test = self.data_loader.data_test[['Positive_Review', 'Negative_Review']]
        self._labels_test = self.data_loader.labels_test

    def rnn(self):
        VOCAB_SIZE = 10000
        encoder_positive = tf.keras.layers.TextVectorization(
            max_tokens=VOCAB_SIZE)
        encoder_positive.adapt(self._data_train['Positive_Review'])

        vocab_positive = np.array(encoder_positive.get_vocabulary())
        print(vocab_positive[:20])

        encoder_negative = tf.keras.layers.TextVectorization(
            max_tokens=VOCAB_SIZE)
        encoder_negative.adapt(self._data_train['Negative_Review'])

        vocab_negative = np.array(encoder_negative.get_vocabulary())
        print(vocab_negative[:20])

        input_positive = tf.keras.Input(shape=(1,), dtype=tf.string)

        input_negative = tf.keras.Input(shape=(1,), dtype=tf.string)

        embedding_positive = encoder_positive(input_positive)
        embedding_positive = tf.keras.layers.Embedding(
            input_dim=len(encoder_positive.get_vocabulary()),
            output_dim=64,
            mask_zero=True)(embedding_positive)
        lstm_positive = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(embedding_positive)

        embedding_negative = encoder_negative(input_negative)
        embedding_negative = tf.keras.layers.Embedding(
            input_dim=len(encoder_negative.get_vocabulary()),
            output_dim=64,
            mask_zero=True)(embedding_negative)
        lstm_negative = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(embedding_negative)

        # input_numeric = tf.keras.Input(shape=(self._data_train.select_dtypes(include=['number']),),dtype=tf.float32)

        concatenated = tf.keras.layers.Concatenate()([lstm_positive, lstm_negative])

        dense1 = tf.keras.layers.Dense(64, activation='relu')(concatenated)
        output = tf.keras.layers.Dense(1)(dense1)

        model = tf.keras.Model(inputs=[input_positive, input_negative], outputs=output)

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(1e-4),
                      metrics=['accuracy'])

        # data_array = np.array(self._data_train, dtype=np.float32)
        history = model.fit([self._data_train['Positive_Review'], self._data_train['Negative_Review']], self._labels_train, epochs=10, validation_split=0.3, shuffle=True)

        # data_test = np.array(self._data_test, dtype=np.float32)
        test_loss, test_acc = model.evaluate([self._data_test['Positive_Review'], self._data_test['Negative_Review']], self._labels_test)

        print('Test Loss:', test_loss)
        print('Test Accuracy:', test_acc)

        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.plot(np.linspace(-3, 3, 100), history, label="Accuracy")
        plt.ylim(None, 1)
        plt.legend()

        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 2)
        plt.plot(np.linspace(-3, 3, 100), history, label="Loss")
        plt.ylim(None, 1)
        plt.legend()

