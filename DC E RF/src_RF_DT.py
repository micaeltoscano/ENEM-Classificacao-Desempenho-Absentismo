import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from matplotlib import pyplot as plt

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def pre_processor_rf_dt(df, objetivo, n_samples = 1_000_000):

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

    df = transformar_colunas_ohe(df)
    df = agregar_questionario(df)

    colunas_q_originais = [c for c in df.columns 
                       if c.startswith('Q') and '_' in c]

    df = df.drop(
                     columns=[
                     'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 
                     'NU_NOTA_MT', 'NU_NOTA_REDACAO', 
                     'TP_LOCALIZACAO_ESC', 'TP_SIT_FUNC_ESC',
                     'TP_PRESENCA_LC', 'TP_PRESENCA_CH',
                     'TP_PRESENCA_CN', 'TP_PRESENCA_MT']
                      + colunas_q_originais)

    return df
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def treinar_rf(x_train, y_train, x_test, y_test, n_estimators, max_depth, max_features, min_samples_split, min_samples_leaf):
    
    rf= RandomForestClassifier(
        n_estimators=n_estimators,        
        max_depth=max_depth,            
        max_features = max_features,     
        min_samples_split = min_samples_split,    
        min_samples_leaf = min_samples_leaf,      
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    rf.fit(x_train, y_train)

    y_pred_train = rf.predict(x_train)
    y_pred_test  = rf.predict(x_test)

    ein  = 1 - accuracy_score(y_train, y_pred_train)
    eout = 1 - accuracy_score(y_test,  y_pred_test)

    print(f"\nEin:  {ein:.4f}")
    print(f"Eout: {eout:.4f}")
    print(f"Gap:  {eout - ein:.4f}  {'overfitting' if eout - ein > 0.05 else 'ok'}")
    print("\n" + classification_report(y_test, y_pred_test))

    return rf
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
def transformar_colunas_ohe(df):
    
    colunas = [
        'Q001','Q002','Q003','Q004','Q006','Q007','Q008','Q009','Q010',
        'Q011','Q012','Q013','Q014','Q015','Q016','Q017','Q018',
        'Q019','Q020','Q021','Q022','Q023','Q024','Q025'
    ]
    
    df = df.dropna(subset=colunas)
    
    df = pd.get_dummies(df, columns=colunas, prefix=colunas, dtype=int)
    
    return df
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
def tune_random_forest(x_train, y_train, n_iter=10, cv=3, scoring='f1_weighted', random_state=42):
    
    n_estimators = [int(x) for x in np.linspace(start=50, stop=100, num=6)]
    max_features = ['sqrt', 'log2']
    max_depth = [int(x) for x in np.linspace(start=10, stop=40, num=4)]
    max_depth.append(None)

    param_grid = {
        'n_estimators':      n_estimators,
        'max_features':      max_features,
        'max_depth':         max_depth,
        'min_samples_split': [10, 20, 50],
        'min_samples_leaf':  [10, 25, 50],
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
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
def transformar_colunas_ohe(df):
    
    colunas = [
        'Q001','Q002','Q003','Q004','Q006','Q007','Q008','Q009','Q010',
        'Q011','Q012','Q013','Q014','Q015','Q016','Q017','Q018',
        'Q019','Q020','Q021','Q022','Q023','Q024','Q025'
    ]
    
    df = df.dropna(subset=colunas)
    
    df = pd.get_dummies(df, columns=colunas, prefix=colunas, dtype=int)
    
    return df
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
def agregar_questionario(df):
    df = df.copy()

    # Q001 — Escolaridade do PAI
    q001_cols = [f'Q001_{l}' for l in 'ABCDEFGH']
    df['escolaridade_pai'] = df[q001_cols].idxmax(axis=1).str.extract(r'_([A-H])')[0]
    df['escolaridade_pai'] = df['escolaridade_pai'].map(
        {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7}
    )
    # Q002 — Escolaridade da MÃE
    q002_cols = [f'Q002_{l}' for l in 'ABCDEFGH']
    df['escolaridade_mae'] = df[q002_cols].idxmax(axis=1).str.extract(r'_([A-H])')[0]
    df['escolaridade_mae'] = df['escolaridade_mae'].map(
        {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7}
    )

    # Escolaridade máxima entre os pais 
    df['escolaridade_pais_max'] = df[['escolaridade_pai','escolaridade_mae']].max(axis=1)

  
    # Q003 — Ocupação do PAI 
    # Q004 — Ocupação da MÃE
    q003_cols = [f'Q003_{l}' for l in 'ABCDEF']
    q004_cols = [f'Q004_{l}' for l in 'ABCDEF']
    df['ocupacao_pai'] = df[q003_cols].idxmax(axis=1).str.extract(r'_([A-F])')[0]
    df['ocupacao_pai'] = df['ocupacao_pai'].map({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5})
    df['ocupacao_mae'] = df[q004_cols].idxmax(axis=1).str.extract(r'_([A-F])')[0]
    df['ocupacao_mae'] = df['ocupacao_mae'].map({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5})

    # Q006 — Renda familiar
    q006_cols = [f'Q006_{l}' for l in 'ABCDEFGHIJKLMNOPQ']
    df['renda_familiar'] = df[q006_cols].idxmax(axis=1).str.extract(r'_([A-Q])')[0]
    df['renda_familiar'] = df['renda_familiar'].map(
        {l:i for i, l in enumerate('ABCDEFGHIJKLMNOPQ')}
    )

    # Q007/Q008 — Bens do domicílio 
    q007_cols = [f'Q007_{l}' for l in 'ABCD']
    q008_cols = [f'Q008_{l}' for l in 'ABCDE']
    df['score_bens_servicos'] = df[q007_cols].sum(axis=1)
    df['score_bens_dom']      = df[q008_cols].sum(axis=1)

    # Q009/Q010/Q011 — Equipamentos (TV, celular, computador, etc)
    
    q009_cols = [f'Q009_{l}' for l in 'ABCDE']
    q010_cols = [f'Q010_{l}' for l in 'ABCDE']
    q011_cols = [f'Q011_{l}' for l in 'ABCDE']
    df['score_equipamentos'] = (
        df[q009_cols].sum(axis=1) +
        df[q010_cols].sum(axis=1) +
        df[q011_cols].sum(axis=1)
    )

    # Q012/Q013/Q014/Q015/Q016/Q017 — Cômodos e estrutura da casa
    q012_cols = [f'Q012_{l}' for l in 'ABCDE']
    q013_cols = [f'Q013_{l}' for l in 'ABCDE']
    df['score_estrutura_casa'] = (
        df[q012_cols].sum(axis=1) +
        df[q013_cols].sum(axis=1)
    )

    # Q024/Q025 — Acesso a computador e internet (capital digital)
    q024_cols = [f'Q024_{l}' for l in 'ABCDE']
    q025_cols = [f'Q025_{l}' for l in 'AB']
    df['acesso_computador'] = df[q024_cols].idxmax(axis=1).str.extract(r'_([A-E])')[0]
    df['acesso_computador'] = df['acesso_computador'].map({'A':0,'B':1,'C':2,'D':3,'E':4})
    df['acesso_internet']   = df[q025_cols].idxmax(axis=1).str.extract(r'_([A-B])')[0]
    df['acesso_internet']   = df['acesso_internet'].map({'A':0,'B':1})

    return df
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------