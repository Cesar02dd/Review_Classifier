import string
import numpy as np
import matplotlib.pyplot as plt
from keras import Model
from keras.layers import Concatenate
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional, GRU, TextVectorization, Input
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model

class DeepLearning:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._data_train = self.data_loader.data_train[['Positive_Review', 'Negative_Review']]
        self._labels_train = self.data_loader.labels_train
        self._data_test = self.data_loader.data_test[['Positive_Review', 'Negative_Review']]
        self._labels_test = self.data_loader.labels_test

    def clean_text(self, text):
        text = text.lower()  # Lowercase the text
        text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
        text = ''.join([char for char in text if not char.isdigit()])  # Remove digits
        text = ' '.join(text.split())  # Remove extra whitespace
        return text

    def _join_reviews(self):
        self._data_train = self._data_train['Positive_Review'] + self._data_train['Negative_Review']
        self._data_test = self._data_test['Positive_Review'] + self._data_test['Negative_Review']

    def rnn(self):
        VOCAB_SIZE = 10000

        encoder_positive = TextVectorization(max_tokens=VOCAB_SIZE, output_sequence_length=None, standardize=None)
        encoder_positive.adapt(self._data_train['Positive_Review'])
        vocab_positive = np.array(encoder_positive.get_vocabulary())
        print(vocab_positive[:20])

        encoder_negative = TextVectorization(max_tokens=VOCAB_SIZE, output_sequence_length=None, standardize=None)
        encoder_negative.adapt(self._data_train['Negative_Review'])
        vocab_negative = np.array(encoder_negative.get_vocabulary())
        print(vocab_negative[:20])

        # Inputs
        input_positive = Input(shape=(1,), dtype=tf.string)
        input_negative = Input(shape=(1,), dtype=tf.string)

        embedding_positive = encoder_positive(input_positive)
        embedding_positive = Embedding(
            input_dim=len(encoder_positive.get_vocabulary()),
            output_dim=64,
            mask_zero=True
        )(embedding_positive)
        lstm_positive = Bidirectional(LSTM(64))(embedding_positive)

        embedding_negative = encoder_negative(input_negative)
        embedding_negative = Embedding(
            input_dim=len(encoder_negative.get_vocabulary()),
            output_dim=64,
            mask_zero=True
        )(embedding_negative)
        lstm_negative = Bidirectional(LSTM(64))(embedding_negative)

        concatenated = Concatenate()([lstm_positive, lstm_negative])

        dense1 = Dense(64, activation='relu')(concatenated)
        output = Dense(1, activation='sigmoid')(dense1)

        model = Model(inputs=[input_positive, input_negative], outputs=output)

        model.summary()
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = model.fit(
            [self._data_train['Positive_Review'], self._data_train['Negative_Review']],
            self._labels_train,
            epochs=10,
            validation_split=0.3,
            shuffle=True
        )

        test_loss, test_acc = model.evaluate(
            [self._data_test['Positive_Review'], self._data_test['Negative_Review']],
            self._labels_test
        )

        print('Test Loss:', test_loss)
        print('Test Accuracy:', test_acc)

        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label="Accuracy")
        plt.plot(history.history['val_accuracy'], label="Val Accuracy")
        plt.ylim(None, 1)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label="Loss")
        plt.plot(history.history['val_loss'], label="Val Loss")
        plt.ylim(None, 1)
        plt.legend()
        plt.show()

    def rnn_3(self):

        VOCAB_SIZE = 10000

        encoder_positive = TextVectorization(max_tokens=VOCAB_SIZE, output_sequence_length=None, standardize=None)
        encoder_positive.adapt(self._data_train['Positive_Review'])
        vocab_positive = np.array(encoder_positive.get_vocabulary())
        print(vocab_positive[:20])

        encoder_negative = TextVectorization(max_tokens=VOCAB_SIZE, output_sequence_length=None, standardize=None)
        encoder_negative.adapt(self._data_train['Negative_Review'])
        vocab_negative = np.array(encoder_negative.get_vocabulary())
        print(vocab_negative[:20])

        # Inputs
        input_positive = Input(shape=(1,), dtype=tf.string)
        input_negative = Input(shape=(1,), dtype=tf.string)

        embedding_positive = encoder_positive(input_positive)
        embedding_positive = Embedding(
            input_dim=len(encoder_positive.get_vocabulary()),
            output_dim=64,
            mask_zero=True
        )(embedding_positive)
        lstm_positive = Bidirectional(LSTM(64))(embedding_positive)

        embedding_negative = encoder_negative(input_negative)
        embedding_negative = Embedding(
            input_dim=len(encoder_negative.get_vocabulary()),
            output_dim=64,
            mask_zero=True
        )(embedding_negative)
        lstm_negative = Bidirectional(LSTM(64))(embedding_negative)

        concatenated = Concatenate()([lstm_positive, lstm_negative])

        dense1 = Dense(64, activation='relu')(concatenated)
        output = Dense(3, activation='softmax')(dense1)  # 3 classes

        model = Model(inputs=[input_positive, input_negative], outputs=output)

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = model.fit(
            [self._data_train['Positive_Review'], self._data_train['Negative_Review']],
            self._labels_train,
            epochs=10,
            validation_split=0.3,
            shuffle=True
        )

        test_loss, test_acc = model.evaluate(
            [self._data_test['Positive_Review'], self._data_test['Negative_Review']],
            self._labels_test
        )

        print('Test Loss:', test_loss)
        print('Test Accuracy:', test_acc)

        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label="Accuracy")
        plt.plot(history.history['val_accuracy'], label="Val Accuracy")
        plt.ylim(None, 1)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label="Loss")
        plt.plot(history.history['val_loss'], label="Val Loss")
        plt.ylim(None, 1)
        plt.legend()
        plt.show()

    def rnn_old(self):
        self._join_reviews()

        # Tokenize the text data
        max_features = 10000  # Maximum number of words to keep based on frequency
        tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')
        tokenizer.fit_on_texts(self._data_train)
        X_train_seq = tokenizer.texts_to_sequences(self._data_train)
        X_test_seq = tokenizer.texts_to_sequences(self._data_test)

        # Pad the sequences
        maxlen = 100  # Maximum length of sequences
        X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen, padding='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen, padding='post')

        # Define the model
        embedding_dim = 100
        model = Sequential([
            Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=maxlen),
            SpatialDropout1D(0.2),
            GRU(64, dropout=0.2, recurrent_dropout=0.2),
            Dense(3, activation='softmax')
        ])

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Define early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Train the model
        epochs = 10
        batch_size = 128  # Increased batch size for faster computation
        history = model.fit(X_train_pad, self._labels_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_test_pad, self._labels_test), callbacks=[early_stopping])

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test_pad, self._labels_test)
        print("Test Loss:", loss)
        print("Test Accuracy:", accuracy)

        # Plotting accuracy and loss
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.title('Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title('Loss')
        plt.show()



    # def rnn_old(self):
    #     VOCAB_SIZE = 10000
    #     encoder_positive = tf.keras.layers.TextVectorization(
    #         max_tokens=VOCAB_SIZE)
    #     encoder_positive.adapt(self._data_train['Positive_Review'])
    #
    #     vocab_positive = np.array(encoder_positive.get_vocabulary())
    #     print(vocab_positive[:20])
    #
    #     encoder_negative = tf.keras.layers.TextVectorization(
    #         max_tokens=VOCAB_SIZE)
    #     encoder_negative.adapt(self._data_train['Negative_Review'])
    #
    #     vocab_negative = np.array(encoder_negative.get_vocabulary())
    #     print(vocab_negative[:20])
    #
    #     input_positive = tf.keras.Input(shape=(1,), dtype=tf.string)
    #
    #     input_negative = tf.keras.Input(shape=(1,), dtype=tf.string)
    #
    #     embedding_positive = encoder_positive(input_positive)
    #     embedding_positive = tf.keras.layers.Embedding(
    #         input_dim=len(encoder_positive.get_vocabulary()),
    #         output_dim=64,
    #         mask_zero=True)(embedding_positive)
    #     lstm_positive = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(embedding_positive)
    #
    #     embedding_negative = encoder_negative(input_negative)
    #     embedding_negative = tf.keras.layers.Embedding(
    #         input_dim=len(encoder_negative.get_vocabulary()),
    #         output_dim=64,
    #         mask_zero=True)(embedding_negative)
    #     lstm_negative = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(embedding_negative)
    #
    #     # input_numeric = tf.keras.Input(shape=(self._data_train.select_dtypes(include=['number']),),dtype=tf.float32)
    #
    #     concatenated = tf.keras.layers.Concatenate()([lstm_positive, lstm_negative])
    #
    #     dense1 = tf.keras.layers.Dense(64, activation='relu')(concatenated)
    #     output = tf.keras.layers.Dense(1)(dense1)
    #
    #     model = tf.keras.Model(inputs=[input_positive, input_negative], outputs=output)
    #
    #     model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #                   optimizer=tf.keras.optimizers.Adam(),
    #                   metrics=['accuracy'])
    #
    #     # data_array = np.array(self._data_train, dtype=np.float32)
    #     history = model.fit([self._data_train['Positive_Review'], self._data_train['Negative_Review']], self._labels_train, epochs=10, validation_split=0.3, shuffle=True)
    #
    #     # data_test = np.array(self._data_test, dtype=np.float32)
    #     test_loss, test_acc = model.evaluate([self._data_test['Positive_Review'], self._data_test['Negative_Review']], self._labels_test)
    #
    #     print('Test Loss:', test_loss)
    #     print('Test Accuracy:', test_acc)
    #
    #     plt.figure(figsize=(16, 8))
    #     plt.subplot(1, 2, 1)
    #     plt.plot(np.linspace(-3, 3, 100), history, label="Accuracy")
    #     plt.ylim(None, 1)
    #     plt.legend()
    #
    #     plt.figure(figsize=(16, 8))
    #     plt.subplot(1, 2, 2)
    #     plt.plot(np.linspace(-3, 3, 100), history, label="Loss")
    #     plt.ylim(None, 1)
    #     plt.legend()

