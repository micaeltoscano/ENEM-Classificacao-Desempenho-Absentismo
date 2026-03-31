import pandas as pd

def pre_processor_inferencia(df):

    df = df.copy()

    df['TP_LOCALIZACAO_ESC']     = df['TP_LOCALIZACAO_ESC'].fillna(0) if 'TP_LOCALIZACAO_ESC' in df.columns else 0
    df['TP_DEPENDENCIA_ADM_ESC'] = df['TP_DEPENDENCIA_ADM_ESC'].fillna(0)
    df['TP_SIT_FUNC_ESC']        = df['TP_SIT_FUNC_ESC'].fillna(0) if 'TP_SIT_FUNC_ESC' in df.columns else 0

    df = transformar_colunas_ohe(df)
    df = agregar_questionario(df)

    colunas_q_originais = [c for c in df.columns if c.startswith('Q') and '_' in c]
    df = df.drop(columns=colunas_q_originais, errors='ignore')

    # Garante ordem igual ao treino
    colunas_modelo = ['Q005', 'TP_FAIXA_ETARIA', 'TP_ESTADO_CIVIL', 'TP_ESCOLA', 'TP_ST_CONCLUSAO', 'IN_TREINEIRO', 'NU_ANO', 'TP_LOCALIZACAO_ESC', 
                      'TP_SIT_FUNC_ESC', 'TP_DEPENDENCIA_ADM_ESC', 'escolaridade_pai', 'escolaridade_mae', 'escolaridade_pais_max', 'ocupacao_pai', 
                      'ocupacao_mae', 'renda_familiar', 'score_bens_servicos', 'score_bens_dom', 'score_equipamentos', 'score_estrutura_casa', 'acesso_computador', 'acesso_internet']

    return df[colunas_modelo]
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def transformar_colunas_ohe(df):
    
    categorias = {
        'Q001': list('ABCDEFGH'),
        'Q002': list('ABCDEFGH'),
        'Q003': list('ABCDEF'),
        'Q004': list('ABCDEF'),
        'Q006': list('ABCDEFGHIJKLMNOPQ'),
        'Q007': list('ABCD'),
        'Q008': list('ABCDE'),
        'Q009': list('ABCDE'),
        'Q010': list('ABCDE'),
        'Q011': list('ABCDE'),
        'Q012': list('ABCDE'),
        'Q013': list('ABCDE'),
        'Q014': list('ABCDE'),
        'Q015': list('ABCDE'),
        'Q016': list('ABCDE'),
        'Q017': list('ABCDE'),
        'Q018': list('ABCDE'),
        'Q019': list('ABCDE'),
        'Q020': list('ABCDE'),
        'Q021': list('ABCDE'),
        'Q022': list('ABCDE'),
        'Q023': list('ABCDE'),
        'Q024': list('ABCDE'),
        'Q025': list('AB'),
    }
    
    colunas = list(categorias.keys())
    df = df.dropna(subset=colunas)
    
    for col, cats in categorias.items():
        df[col] = pd.Categorical(df[col], categories=cats)
    
    df = pd.get_dummies(df, columns=colunas, prefix=colunas, dtype=int)
    
    return df

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


