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
from src import *

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
def treinar_por_cor(df, cor, n_estimators, max_depth, max_features, min_samples_split, min_samples_leaf):
    
    df_cor = df[df['COR_LC'] == cor].copy()
    df_cor = df_cor.dropna()
    
    df_cor['CLASSE'] = df_cor.groupby('NU_ANO')['TOTAL_ACERTOS_GERAL'].transform(
        lambda x: pd.qcut(x, q=3, labels=[1, 2, 3])
    ).astype('Int64')
    
    df_cor = df_cor.dropna(subset=['CLASSE'])
    df_cor['CLASSE'] = df_cor['CLASSE'].astype(int)
    
    features = (
        [f'ACERTO_LC_{i:02d}' for i in range(1, 46)] +
        [f'ACERTO_CH_{i:02d}' for i in range(1, 46)] +
        [f'ACERTO_CN_{i:02d}' for i in range(1, 46)] +
        [f'ACERTO_MT_{i:02d}' for i in range(1, 46)] +
        ['TOTAL_ACERTOS_LC', 'TOTAL_ACERTOS_CH', 'TOTAL_ACERTOS_CN', 'TOTAL_ACERTOS_MT']
    )
    
    X = df_cor[features]
    y = df_cor['CLASSE']
    
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f'\n=== {cor} ===')
    return treinar_rf(x_train, y_train, x_test, y_test,
                      n_estimators, max_depth, max_features,
                      min_samples_split, min_samples_leaf)

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
def calcular_acertos_por_area(df):
    cols_questao = [c for c in df.columns if c.startswith('questao_')]
    
    for sigla in ['LC', 'MT', 'CH', 'CN']:
        cols_area = [c for c in cols_questao if c.endswith(f'_{sigla}')]
        df[f'ACERTOS_{sigla}'] = df[cols_area].sum(axis=1)
    
    return df
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
def tune_random_forest(x_train, y_train, x_test, y_test, n_iter=10, cv=3, scoring='f1_weighted', random_state=42):
    
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