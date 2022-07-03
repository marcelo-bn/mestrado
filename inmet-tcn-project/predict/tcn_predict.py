import os, sys
os.chdir("..")
sys.path.append(str(os.path.abspath(os.curdir)+'/format'))
from format import Format


class TCNpredict:
    '''
    Classe responsalvel pela predicao 
    utilizando modelos TCN
    '''
    def __init__(self):
        pass

    def tcn_predict_two_days(col, data, n, m, model):
        '''
          Previsao Dois Dias

          col: variavel em analise
          data: dataframe completo
          n: amostras a serem previstas
          m: amostras necessarias para o modelo
          model: modelo treinado

          retorna uma lista com n amostras previstas. Para
          esta aplicacao n sera sempre igual a 12. Ou seja,
          cada modelo retornará 12 amostra, ou a previsao
          para 2 dias.

      '''
        # Lista de m valores reais
        train_data = data[col][-m:].values

        # Formatando dados
        f = Format()
        r = f.format_array([train_data])
        data_formatted = r[0]

        forecast = []
        list_prev = []

        for j in range(n):
            if len(list_prev) == 0:
                value = model.predict(data_formatted)
                value = value[0, 0].item()
                list_prev.append(value)
            else:
                data_formatted = f.new_data(data_formatted, list_prev)
                value = model.predict(data_formatted)
                value = value[0, 0].item()
                list_prev.append(value)

        forecast.append(list_prev)

        return forecast

    def tcn_predict_next_hour(col, data, n, m, model):
        '''
        Previsao Proxima Hora

        col: variavel em analise
        data: dataframe completo
        n: amostras a serem previstas
        m: amostras necessarias para o modelo
        model: modelo treinado

        '''
        forecast = []
        list_prev = []

        # Formatando dados
        f = Format()
        r = f.format_array([data[:m]])
        data_formatted = r[0]

        for j in range(n):
            if len(list_prev) == 0:
                value = model.predict(data_formatted)
                value = value[0, 0].item()
                list_prev.append(value)
            else:
                data_formatted = f.new_data(data_formatted, [data[m + j]])
                value = model.predict(data_formatted)
                value = value[0, 0].item()
                list_prev.append(value)

        # Armazenando as 24 primeiras amostras
        t = data[:m]
        for i in reversed(t):
            list_prev.insert(0, i)

        forecast.append(list_prev)

        return forecast
