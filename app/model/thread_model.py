import pandas as pd
from .data_loader import DataLoader
from aurora import text_preprocessing
import logging
from keras.optimizers import Adadelta
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout, Activation, Lambda, merge, Input, TimeDistributed, Convolution1D, MaxPooling2D, \
    Embedding, Concatenate, LSTM
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.models import Model, load_model
import pickle
import numpy as np
from time import time
import os

LEVELS = {'debug': logging.DEBUG,
          'info': logging.INFO,
          'warning': logging.WARNING,
          'error': logging.ERROR,
          'critical': logging.CRITICAL}


class SentimentModel:
    setting = {'MAX_NUM_WORDS': 9548, 'MAX_SEQUENCE_LENGTH': 400,
               'EMBEDDING_DIM': 100, 'BATCH_SIZE': 15, 'EPOCHS': 10, 'NB_FILTER': 150, 'HIDDEN_DIM': 100}

    def __init__(self, log_level='info'):
        logging.basicConfig(level=LEVELS[log_level])
        # Create folder if not exist
        os.makedirs("app/resources/model", exist_ok=True)
        self.log = logging.getLogger(self.__class__.__name__)
        self.data_loader = DataLoader()
        self.train, self.test = self.data_loader.load_data()

    def train_model(self):
        self.preprocessing_data()
        self.tokenize()
        self.set_model()

    def preprocessing_data(self):
        self.log.info(f"Data before preprocess len(train)={len(self.train)}, len(test)={len(self.test)}")
        self.train['message'] = self.train['message'].apply(lambda x: text_preprocessing(x))
        self.test['message'] = self.test['message'].apply(lambda x: text_preprocessing(x))
        # Drop invalid data
        self.train = self.train[self.train['message'] != ""]
        self.test = self.test[self.test['message'] != ""]
        self.log.info(f"Data after preprocess len(train)={len(self.train)}, len(test)={len(self.test)}")

        classes = list(pd.get_dummies(self.train['label']).columns)
        with open('app/resources/model/classes.pickle', 'wb') as handle:
            pickle.dump(classes, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.X_train, self.y_train, self.X_test, self.y_test = self.train['message'], pd.get_dummies(
            self.train['label']).values, self.test[
                                                                   'message'], pd.get_dummies(self.test['label']).values

    def tokenize(self):
        self.tokenizer = Tokenizer(num_words=self.setting['MAX_NUM_WORDS'], char_level=False,
                                   filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n')
        self.tokenizer.fit_on_texts(self.X_train)

        with open('app/resources/model/tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.X_train_sequences = self.tokenizer.texts_to_sequences(self.X_train)
        self.X_test_sequences = self.tokenizer.texts_to_sequences(self.X_test)
        train_seq_lens = [len(s) for s in self.X_train_sequences]
        test_seq_lens = [len(s) for s in self.X_test_sequences]
        self.log.info("Train: average length: %0.1f" % np.mean(train_seq_lens))
        print("Train: max length: %d" % max(train_seq_lens))
        self.log.info("Test: average length: %0.1f" % np.mean(test_seq_lens))
        print("Test: max length: %d" % max(test_seq_lens))

        self.X_train_dim = pad_sequences(self.X_train_sequences, self.setting['MAX_SEQUENCE_LENGTH'])
        print('Shape of data tensor:', self.X_train_dim.shape)

    def set_model(self):
        self.input_layer = Input(shape=(self.setting['MAX_SEQUENCE_LENGTH'],), dtype='int32', name='main_input')
        self.emb_layer = Embedding(self.setting['MAX_NUM_WORDS'],
                                   self.setting['EMBEDDING_DIM'],
                                   input_length=self.setting['MAX_SEQUENCE_LENGTH']
                                   )(self.input_layer)
        con3_layer = Convolution1D(nb_filter=self.setting['NB_FILTER'],
                                   filter_length=3,
                                   border_mode='valid',
                                   activation='relu',
                                   subsample_length=1)(self.emb_layer)

        pool_con3_layer = Lambda(lambda x: K.max(x, axis=1), output_shape=(self.setting['NB_FILTER'],))(
            con3_layer)

        # số filter của convolution layer = 5
        con4_layer = Convolution1D(nb_filter=self.setting['NB_FILTER'],
                                   filter_length=5,
                                   border_mode='valid',
                                   activation='relu',
                                   subsample_length=1)(self.emb_layer)

        pool_con4_layer = Lambda(lambda x: K.max(x, axis=1), output_shape=(self.setting['NB_FILTER'],))(con4_layer)

        # số filter của convolution layer = 7
        con5_layer = Convolution1D(nb_filter=self.setting['NB_FILTER'],
                                   filter_length=7,
                                   border_mode='valid',
                                   activation='relu',
                                   subsample_length=1)(self.emb_layer)

        pool_con5_layer = Lambda(lambda x: K.max(x, axis=1), output_shape=(self.setting['NB_FILTER'],))(con5_layer)

        # Nối các Convolution layer
        multiply_layer = Concatenate()
        cnn_layer = multiply_layer([pool_con3_layer, pool_con5_layer, pool_con4_layer])

        # LSTM
        x = Embedding(self.setting['MAX_NUM_WORDS'], self.setting['EMBEDDING_DIM'],
                      input_length=self.setting['MAX_SEQUENCE_LENGTH'])(self.input_layer)
        lstm_layer = LSTM(128)(x)

        # Nối các Convolution layer với lstm layer
        cnn_lstm_layer = multiply_layer([lstm_layer, cnn_layer])

        dense_layer = Dense(self.setting['HIDDEN_DIM'] * 2, activation='sigmoid')(cnn_lstm_layer)
        output_layer = Dropout(0.2)(dense_layer)
        output_layer = Dense(3, trainable=True, activation='softmax')(output_layer)

        # Khởi tạo model dùng function API
        model = Model(input=[self.input_layer], output=[output_layer])

        # Tối ưu hóa model
        adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)

        # Biên tập lại toàn bộ model đã build
        model.compile(loss='categorical_crossentropy',
                      optimizer="adamax",
                      metrics=['accuracy'])
        self.log.info(model.summary())

        callbacks = [
            ReduceLROnPlateau(), EarlyStopping(patience=4), \
            ModelCheckpoint(filepath='app/resources/model/model.h5', save_best_only=True)
        ]
        self.log.info('Train...')

        # Đưa data vào training để tìm tham số model
        self.y_train = np.array(self.y_train)
        self.log.info(f"X shape: {self.X_train_dim.shape}")
        self.log.info(f"Y shape: {self.y_train.shape}")

        model.fit(self.X_train_dim, self.y_train,
                  batch_size=self.setting['BATCH_SIZE'],
                  epochs=self.setting['EPOCHS'],
                  callbacks=callbacks, validation_split=0.1)

    def load_classes(self):
        with open('app/resources/model/classes.pickle', 'rb') as handle:
            classes = pickle.load(handle)
        self.classes = classes

    def load_saved_model(self):
        return load_model('app/resources/model/model.h5')

    def load_tokenizer(self):
        with open('app/resources/model/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        self.tokenizer = tokenizer

    def predict(self, text):
        result = {}
        if getattr(self, "tokenizer", None) is None:
            self.load_tokenizer()
        if getattr(self, "model", None) is None:
            self.load_saved_model()
        if getattr(self, "classes", None) is None:
            self.load_classes()
        start = time()
        text = text_preprocessing(text)
        sequences = self.tokenizer.texts_to_sequences([text])
        sequences_input = pad_sequences(sequences, maxlen=self.setting['MAX_SEQUENCE_LENGTH'])
        result['preprocess_time'] = time() - start
        # Recommend
        start = time()
        predictions = self.model.predict(sequences_input)
        pred_index = np.argmax(predictions, axis=1)
        result['pred_time'] = time() - start
        result['label'] = self.classes[int(pred_index)]
        return result
