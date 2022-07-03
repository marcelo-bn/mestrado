import os
import sys
import math
import pandas as pd

from tcn import TCN, tcn_full_summary
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model

os.chdir("..")
sys.path.append(str(os.path.abspath(os.curdir)+'/format'))

from format import Format

class TCNmodel():
    '''
    Classe responsavel por gerar e manipular
    modelos de previsao TCN.
    '''
    def __init__(self):
        pass

    def model_generator_dois_dias(self, df, col, lt, m):
        ''' Gera o modelo de Dois Dias'''
        model = Sequential()
        f = Format()

        # Normalizacao
        max_value = df[col].max()
        min_value = df[col].min()
        df[col] = (df[col] - min_value) / (max_value - min_value)

        # Dataset de treino e teste
        index_test = math.ceil(len(df.index) * lt)
        train_set = df[:-48]
        test_set = df[-48:]

        # Criando datasets de treino e teste de acordo com o valor de h (num. amostras para previsao)
        train_X, train_y = f.dataframe_to_Xy(train_set[col], m);
        test_X, test_y = f.dataframe_to_Xy(test_set[col], m);

        # Treinando modelo
        model.add(TCN(units=64, input_shape=(m, 1), activation="relu", recurrent_activation="sigmoid"))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        historico = model.fit(train_X, train_y, epochs=3, batch_size=1, verbose=1)

        return train_set, test_set, min_value, max_value, model

    def model_generator_proxima_hora(self, df, col, lt, m):
        ''' Gera o modelo de Proxima Hora'''
        model = Sequential()
        f = Format()

        # Normalizacao
        max_value = df[col].max()
        min_value = df[col].min()
        df[col] = (df[col] - min_value) / (max_value - min_value)

        # Dataset de treino e teste
        index_test = math.ceil(len(df.index) * lt)
        train_set = df[:-index_test]
        test_set = df[-index_test:]

        # Criando datasets de treino e teste de acordo com o valor de h (num. amostras para previsao)
        train_X, train_y = f.dataframe_to_Xy(train_set[col], m)
        test_X, test_y = f.dataframe_to_Xy(test_set[col], m)

        # Treinando modelo
        model.add(TCN(units=64, input_shape=(m, 1), activation="relu", recurrent_activation="sigmoid"))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        history = model.fit(train_X, train_y, epochs=3, batch_size=1, verbose=1)

        return train_set, test_set, min_value, max_value, model

    def model_save(self, train, test, min_value, max_value, model, path):
        ''' Salva modelo '''
        try:
            model.save(path)
            f = Format()
            f.dataframe_to_csv(train, path + '/train_set.csv')
            f.dataframe_to_csv(test, path + '/test_set.csv')
            norm_df = pd.DataFrame({'min_value': [min_value], 'max_value': [max_value]})
            f.dataframe_to_csv(norm_df, path + '/norm.csv')
            return 1
        except Exception as e:
            print('> Erro ao salvar modelo!')
            return 0

    ''' Carrega modelo '''

    def load_model(self, path):

        try:
            model = load_model(path)
            train_set = pd.read_csv(path + '/train_set.csv', index_col='Data', parse_dates=True, low_memory=False)
            test_set = pd.read_csv(path + '/test_set.csv', index_col='Data', parse_dates=True, low_memory=False)
            norm = pd.read_csv(path + '/norm.csv')
            min_value = norm['min_value'].values[0]
            max_value = norm['max_value'].values[0]
            return train_set, test_set, min_value, max_value, model
        except Exception as e:
            print('> Erro ao carregar modelo!')
            return 0
