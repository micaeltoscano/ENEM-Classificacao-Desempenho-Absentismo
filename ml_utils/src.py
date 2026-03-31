import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def transformar_colunas_ohe(df):
    
    colunas = [
        'Q001','Q002','Q003','Q004','Q006','Q007','Q008','Q009','Q010',
        'Q011','Q012','Q013','Q014','Q015','Q016','Q017','Q018',
        'Q019','Q020','Q021','Q022','Q023','Q024','Q025'
    ]
    
    df = df.dropna(subset=colunas)
    
    df = pd.get_dummies(df, columns=colunas, prefix=colunas, dtype=int)
    
    return df
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def agregar_questionario(df):

    df = df.copy()
    
    q001_cols = [f'Q001_{l}' for l in 'ABCDEFGH']
    df['escolaridade_pai'] = df[q001_cols].idxmax(axis=1).str.extract(r'_([A-H])')[0]
    df['escolaridade_pai'] = df['escolaridade_pai'].map(
        {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7}
    )
    
    q002_cols = [f'Q002_{l}' for l in 'ABCDEFGH']
    df['escolaridade_mae'] = df[q002_cols].idxmax(axis=1).str.extract(r'_([A-H])')[0]
    df['escolaridade_mae'] = df['escolaridade_mae'].map(
        {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7}
    )

    df['escolaridade_pais_max'] = df[['escolaridade_pai','escolaridade_mae']].max(axis=1)

  
    q003_cols = [f'Q003_{l}' for l in 'ABCDEF']
    q004_cols = [f'Q004_{l}' for l in 'ABCDEF']
    df['ocupacao_pai'] = df[q003_cols].idxmax(axis=1).str.extract(r'_([A-F])')[0]
    df['ocupacao_pai'] = df['ocupacao_pai'].map({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5})
    df['ocupacao_mae'] = df[q004_cols].idxmax(axis=1).str.extract(r'_([A-F])')[0]
    df['ocupacao_mae'] = df['ocupacao_mae'].map({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5})

    q006_cols = [f'Q006_{l}' for l in 'ABCDEFGHIJKLMNOPQ']
    df['renda_familiar'] = df[q006_cols].idxmax(axis=1).str.extract(r'_([A-Q])')[0]
    df['renda_familiar'] = df['renda_familiar'].map(
        {l:i for i, l in enumerate('ABCDEFGHIJKLMNOPQ')}
    )

    q007_cols = [f'Q007_{l}' for l in 'ABCD']
    q008_cols = [f'Q008_{l}' for l in 'ABCDE']
    df['score_bens_servicos'] = df[q007_cols].sum(axis=1)
    df['score_bens_dom']      = df[q008_cols].sum(axis=1)

    
    q009_cols = [f'Q009_{l}' for l in 'ABCDE']
    q010_cols = [f'Q010_{l}' for l in 'ABCDE']
    q011_cols = [f'Q011_{l}' for l in 'ABCDE']
    df['score_equipamentos'] = (
        df[q009_cols].sum(axis=1) +
        df[q010_cols].sum(axis=1) +
        df[q011_cols].sum(axis=1)
    )

    q012_cols = [f'Q012_{l}' for l in 'ABCDE']
    q013_cols = [f'Q013_{l}' for l in 'ABCDE']
    df['score_estrutura_casa'] = (
        df[q012_cols].sum(axis=1) +
        df[q013_cols].sum(axis=1)
    )

    q024_cols = [f'Q024_{l}' for l in 'ABCDE']
    q025_cols = [f'Q025_{l}' for l in 'AB']
    df['acesso_computador'] = df[q024_cols].idxmax(axis=1).str.extract(r'_([A-E])')[0]
    df['acesso_computador'] = df['acesso_computador'].map({'A':0,'B':1,'C':2,'D':3,'E':4})
    df['acesso_internet']   = df[q025_cols].idxmax(axis=1).str.extract(r'_([A-B])')[0]
    df['acesso_internet']   = df['acesso_internet'].map({'A':0,'B':1})

    return df
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def preparar_dados(df, objetivo, n_samples = 50_000):

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

    df['TP_LOCALIZACAO_ESC'] = df['TP_LOCALIZACAO_ESC'].fillna(0)
    df['TP_DEPENDENCIA_ADM_ESC'] = df['TP_DEPENDENCIA_ADM_ESC'].fillna(0)
    df['TP_SIT_FUNC_ESC'] = df['TP_SIT_FUNC_ESC'].fillna(0)

    df = df.dropna(subset=[f'Q{i:03d}' for i in range(1, 26)])

    df = transformar_colunas_ohe(df)
    df = agregar_questionario(df)

    colunas_q_originais = [c for c in df.columns 
                       if c.startswith('Q') and '_' in c]

    df = df.drop(
                     columns=[
                     'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 
                     'NU_NOTA_MT', 'NU_NOTA_REDACAO', 
                     'TP_PRESENCA_LC', 'TP_PRESENCA_CH',
                     'TP_PRESENCA_CN', 'TP_PRESENCA_MT']
                      + colunas_q_originais)

    return df
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def buscar_hiperparametros_rf(x_train, y_train, n_iter=10, cv=5, scoring='f1_weighted', random_state=42):
    
    max_depth = [int(x) for x in np.linspace(start=10, stop=40, num=4)]
    max_depth.append(None)
    
    param_grid = {
    'n_estimators':      [100, 200, 300],
    'max_features':      ['sqrt', 'log2', 0.3],  
    'max_depth':         [5, 10, 15, 20],        
    'min_samples_split': [20, 50, 100],            
    'min_samples_leaf':  [20, 50, 100],            
    'max_samples':       [0.6, 0.8, 1.0],         
}

    rf = RandomForestClassifier(class_weight='balanced', random_state=random_state)
    
    cv_rf = RandomizedSearchCV(     
        estimator=rf,
        param_distributions=param_grid, 
        n_iter=n_iter,              
        cv=cv,
        scoring=scoring,
        verbose=2,
        n_jobs=-1,
        random_state=random_state    
    )

    cv_rf.fit(x_train, y_train)

    return cv_rf
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def pre_processor(X_train):
    
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
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def create_model(input_dim, neurons, learning_rate, l2_reg, dropout):

    model = Sequential()

    model.add(Input(shape=(input_dim,)))

    model.add(Dense(neurons,
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    activation='relu'))
    if dropout > 0:
        model.add(Dropout(dropout))
        
    model.add(Dense(1, kernel_initializer='he_normal', activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    return model
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def num_max_neuronio(X, d):
    CT = len(X)
    return int((CT - 10)/(10 * (d + 2)))
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def dt_amostras_maiores():
    
    df = preparar_dados(df, objetivo = 'Desempenho', n_samples = 50_000)

    X = df.drop(columns=['MEDIA', 'FALTOU'])
    y_media = df['MEDIA']

    X_train, X_test, y_train, y_test = train_test_split(X, y_media, test_size=0.2, random_state=42)

    quantil = y_train.quantile(0.5)
    y_train = (y_train >= quantil).astype(int)
    y_test  = (y_test  >= quantil).astype(int)

    clf = DecisionTreeClassifier(**CV_clf.best_params_)
    clf.fit(X_train, y_train)

    print('Ein: %0.4f' % (1 - accuracy_score(y_train, CV_clf.predict(X_train))))
    print('Eval: %0.4f' % (1 - accuracy_score(y_test, CV_clf.predict(X_test))))

    print(classification_report(y_test, clf.predict(X_test)))
