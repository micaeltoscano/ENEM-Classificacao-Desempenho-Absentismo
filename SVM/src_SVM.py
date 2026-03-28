import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def pre_processor_SVM(df, objetivo, n_samples = 50000):

    df = df.sample(n_samples, random_state=42)

    FALTOU = (
    (df['TP_PRESENCA_CH'] != 1) | 
    (df['TP_PRESENCA_LC'] != 1) | 
    (df['TP_PRESENCA_CN'] != 1) | 
    (df['TP_PRESENCA_MT'] != 1)
)

    df['FALTOU'] = FALTOU.astype(int)

    if objetivo == 'Desempenho':

        df = df[df['FALTOU'] == 0]

        df['MEDIA'] = (df['NU_NOTA_CN'] + df['NU_NOTA_CH'] + df['NU_NOTA_MT']+  df['NU_NOTA_LC'] + df['NU_NOTA_REDACAO']) / 5

        df['CLASSE'] = df.groupby('NU_ANO')['MEDIA'].transform(lambda x: pd.qcut(x, q=2, labels=[0,1])).astype('Int64')

        df['CLASSE'] = df['CLASSE'].astype(int)

    df = df[df['TP_ESCOLA'].isin([2,3])]
    df = df[df['TP_ESTADO_CIVIL'].isin([1,2,3,4])]

    df['TP_LOCALIZACAO_ESC'] = df['TP_LOCALIZACAO_ESC'].fillna(0)
    df['TP_DEPENDENCIA_ADM_ESC'] = df['TP_DEPENDENCIA_ADM_ESC'].fillna(0)
    df['TP_SIT_FUNC_ESC'] = df['TP_SIT_FUNC_ESC'].fillna(0)

    df = df.dropna(subset=[f'Q{i:03d}' for i in range(1, 26)])

    df = df.drop(
                     columns=[
                     'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 
                     'NU_NOTA_MT', 'NU_NOTA_REDACAO',
                     'TP_PRESENCA_LC', 'TP_PRESENCA_CH',
                     'TP_PRESENCA_CN', 'TP_PRESENCA_MT']
                     )

    return df

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def transformar(X_train):
    
    nominais = ['TP_ESCOLA', 'TP_DEPENDENCIA_ADM_ESC',
                'TP_ESTADO_CIVIL', 'TP_LOCALIZACAO_ESC', 'TP_SIT_FUNC_ESC']
    ordinais = ['TP_FAIXA_ETARIA', 'TP_ST_CONCLUSAO']
    binarias = ['IN_TREINEIRO']
    questionario = [f'Q{str(i).zfill(3)}' for i in range(1, 26) if i != 5]

    categorias_quest = []
    for i in range(1, 26):
        if i == 5:
            continue
        elif i == 6:
            categorias_quest.append(list('ABCDEFGHIJKLMNOPQ'))
        elif i == 25:
            categorias_quest.append(list('AB'))
        elif i in [1, 2]:
            categorias_quest.append(list('ABCDEFGH'))
        elif i in [3, 4]:
            categorias_quest.append(list('ABCDEF'))
        else:
            categorias_quest.append(list('ABCDE'))

    preprocessor = ColumnTransformer(transformers=[
        ('nominal',      OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominais),
        ('ordinal',      OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinais),
        ('questionario', OrdinalEncoder(categories=categorias_quest, handle_unknown='use_encoded_value', unknown_value=-1), questionario),
        ('binaria',      'passthrough', binarias),
    ], remainder='drop')

    preprocessor.fit(X_train)

    return preprocessor
