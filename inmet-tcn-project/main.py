'''
    Arquivo principal para o script de
    teste/estudo com dados INMET e algoritmo
    TCN.
'''

'''' Python imports  '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tcn import TCN, tcn_full_summary
from keras.layers import Dense
from keras.models import Sequential

'''' Own modules imports  '''
from predict.tcn_predict import TCNpredict
from models.tcn_model import TCNmodel
from format.tcn_format import TCNformat

