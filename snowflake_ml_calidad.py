#!/usr/bin/env python3
"""ML en Snowflake: clasifica CALIDAD de recepcion (ESTANDAR vs CONDICIONADO).
Corre en Notebooks Snowflake (usa get_active_session) o local (connector + env).
Campeon: PCA(95%) + LogReg vs RF base. Train/val/test por columna SPLIT.
Opcional: escribe predicciones a PRED_CALIDAD_ML. Imprime REPORTE.
"""
import os
SEED = 42
NUM = ['MES_COS', 'MES_SEN', 'TEMP_C_7D', 'LLUVIA_MM_7D', 'SUPERFICIE_HA',
       'DIAS_DESDE_CONTRATO', 'OBJETIVO_HUMEDAD_PCT', 'PRECIO_ACORDADO_USD_TN',
       'HUMEDAD_RELATIVA_PCT_7D', 'VOLUMEN_COMPROMETIDO_TN']
CAT = ['TIPO_PRODUCTOR']
def cargar():
    cols = NUM + CAT + ['SPLIT', 'OBJETIVO_CALIDAD', 'ID_RECEPCION']
    try:
        from snowflake.snowpark.context import get_active_session
    except ImportError:
        return _cargar_local(cols)  # fuera de Snowflake: aqui si pide credenciales
    session = get_active_session()  # dentro de Snowflake: SIN credenciales
    print('Origen: sesion Snowflake (Snowpark), sin credenciales')
    df = session.sql(f"SELECT {', '.join(cols)} FROM ARQUI1.CAPA_BRONCE.CALIDAD_RECEPCIONES").to_pandas()
    return df, session


def _cargar_local(cols):
    import snowflake.connector
    pwd = os.getenv('SNOWFLAKE_PASSWORD', '').strip().strip('\'"')
    if not pwd:
        raise SystemExit("Sin password: export SNOWFLAKE_PASSWORD='clave'")
    kw = dict(user=os.getenv('SNOWFLAKE_USER', 'ENRIQUE'), password=pwd,
              account=os.getenv('SNOWFLAKE_ACCOUNT', 'WMNAMCT-LN60692'),
              database=os.getenv('SNOWFLAKE_DB', 'ARQUI1'),
              schema=os.getenv('SNOWFLAKE_SCHEMA', 'CAPA_BRONCE'),
              warehouse=os.getenv('SNOWFLAKE_WH', 'COMPUTE_WH'), login_timeout=60)
    if os.getenv('SNOWFLAKE_ROLE'):
        kw['role'] = os.getenv('SNOWFLAKE_ROLE')
    c = snowflake.connector.connect(**kw)
    cur = c.cursor(); cur.execute(f"SELECT {', '.join(cols)} FROM CALIDAD_RECEPCIONES")
    import pandas as pd
    print('Origen: connector local')
    return pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description]), None
def main():
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                 roc_auc_score, average_precision_score, confusion_matrix)
    df, session = cargar()
    df.columns = df.columns.str.lower()
    num = [x.lower() for x in NUM]
    cat = [x.lower() for x in CAT]
    df['y'] = (df['objetivo_calidad'] == 'CONDICIONADO').astype(int)
    def _pre():
        return ColumnTransformer([
            ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), num),
            ('cat', make_pipeline(SimpleImputer(strategy='constant', fill_value='FALTA'),
                                   OneHotEncoder(handle_unknown='ignore')), cat)])
    tr, va, te = df[df['split'] == 'train'], df[df['split'] == 'validation'], df[df['split'] == 'test']
    Xtr, ytr, Xva, yva, Xte, yte = tr[num + cat], tr['y'], va[num + cat], va['y'], te[num + cat], te['y']
    print(f'train={len(tr)}(+{int(ytr.sum())}) val={len(va)}(+{int(yva.sum())}) test={len(te)}(+{int(yte.sum())})')
    lr = dict(max_iter=1000, class_weight='balanced', random_state=SEED)
    rf = dict(n_estimators=300, min_samples_leaf=5, class_weight='balanced_subsample', n_jobs=-1, random_state=SEED)
    cand = {
        'RF_base': make_pipeline(_pre(), RandomForestClassifier(**rf)),
        'PCA+LogReg': make_pipeline(_pre(), PCA(n_components=0.95, random_state=SEED), LogisticRegression(**lr)),
    }
    best, bn = -1, ''
    for n, m in cand.items():
        m.fit(Xtr, ytr)
        p, s = m.predict(Xva), m.predict_proba(Xva)[:, 1]
        f1 = f1_score(yva, p, zero_division=0)
        print(f'{n}: F1={f1:.3f} AUC={roc_auc_score(yva, s):.3f} rec={recall_score(yva, p):.3f}')
        if f1 > best:
            best, bn = f1, n
    m = cand[bn]
    p, s = m.predict(Xte), m.predict_proba(Xte)[:, 1]
    print('=== REPORTE (TEST unico) ===')
    print(f'Modelo={bn} acc={accuracy_score(yte, p):.3f} prec={precision_score(yte, p, zero_division=0):.3f} '
          f'rec={recall_score(yte, p):.3f} F1={f1_score(yte, p, zero_division=0):.3f} '
          f'AUC={roc_auc_score(yte, s):.3f} AP={average_precision_score(yte, s):.3f}')
    print('Matriz [[TN FP][FN TP]]:', confusion_matrix(yte, p).tolist())
    if os.getenv('ESCRIBIR', '0') == '1':
        out = te[['id_recepcion']].copy()
        out['pred'] = p
        out['score'] = np.round(s, 4)
        out.columns = [c.upper() for c in out.columns]
        if session is not None:
            session.write_pandas(out, 'PRED_CALIDAD_ML', auto_create_table=True, overwrite=True)
        else:
            out.to_csv('data/pred_calidad_ml.csv', index=False, encoding='utf-8')
        print('Predicciones guardadas (PRED_CALIDAD_ML o CSV)')
if __name__ == '__main__':
    main()
