import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
#from sklearn.metrics import mean_absolute_error, mean_squared_error


class Format:
    '''
    Classe responsavel pela formatacao de dados.
    '''

    def __init__(self):
        pass

    def teste(self):
        return 'worked!'

    def dataframe_to_Xy(self, df, n_in):
        '''Cria os datasets baseado no valor de n_in'''
        df_np = df
        X, y = [], []
        for i in range(len(df) - n_in):
            X.append([[a] for a in df_np[i:i + n_in]])
            y.append(df_np[i + n_in])
        return np.array(X), np.array(y)

    def dataframe_to_csv(self, dataframe, path):
        ''' Converte dataframe em csv '''
        dataframe['Data'] = dataframe.index
        dataframe.to_csv(path, index=False)

    def format_array(self, data):
        '''
        Formatacao dos dados que serao utilizados
        na previsao [[a,b,c],[d,e,f]] -> [[[[a],[b],[c]]],[[[d],[e],[f]]]] ->
        format[0] = [[[a],[b],[c]]]
        '''
        format = []
        for d in data:
            format.append([[[a] for a in d]])

        return format

    def new_data(self, data, new_data):
        '''
        Atualiza vetor de dados para previsao
        com base no ultimo valor previsto.
        data = [[[a],[b],[c],[d],[e]]]
        new_data = [f]
        result = [[[b],[c],[d],[e],[f]]]
        '''
        new_array_data = []
        new_array_data_shift = []
        l1 = data[0]

        for i in range(len(l1)):
            a = l1[i]
            new_array_data.append(a[0])

        for i in range(len(new_array_data) - 1):
            new_array_data_shift.append(new_array_data[i + 1])

        new_array_data_shift.append(new_data[-1])

        res = self.format_array([new_array_data_shift])

        return res[0]

    def plot_and_errors(self, forecast, real, forecast_norm, real_norm, title, ax, ay, x_label, y_label):
        # ''' Plot de graficos e calculo de erros'''
        # plt.rcParams["figure.figsize"] = (ax, ay)
        #
        # mae = mean_absolute_error(real, forecast)
        # rmse = math.sqrt(mean_squared_error(real, forecast))
        #
        # mape = np.mean(np.abs((real - forecast) / real)) * 100
        # maape = np.mean(np.arctan(np.abs((real_norm - forecast_norm) / (real_norm)))) * 100
        #
        # print("Mean Absolute Error:", mae)
        # print("Root Mean Square Error:", rmse)
        # print("Mean Absolute Error Percentage:", mape)
        # print("Mean Arctangent Absolute Percentage Error:", round(maape, 3))
        #
        # plt.plot(real)
        # plt.plot(forecast)
        # plt.title(title)
        # plt.xlabel(x_label)
        # plt.ylabel(y_label)
        # plt.legend(['Dados reais', 'Previsão'])
        pass


    