#!/usr/bin/env python3
"""ML multiclase en Snowflake: grado CALIDAD_GRANO (ESTANDAR/PREMIUM/DESCUENTO).
1.55M recepciones. LogReg multinomial vs HGB (nulos nativos) vs PCA+LogReg.
Split estratificado 70/15/15. Seleccion en val (F1-macro), test unico.
Uso en Snowflake (sin credenciales) o local. SAMPLE_PCT=20 por defecto.
ESCRIBIR=1 guarda PRED_CALIDAD_MULTI. Imprime REPORTE.
"""
import os
SEED = 42
NUM = ['PESO_NETO', 'HUMEDAD_PCT', 'IMPUREZAS_PCT', 'ACEITE_PCT', 'PROTEINA_PCT',
       'TEMPERATURA_GRANO_C', 'MES']
CAT = ['ID_SILO']
def cargar():
    sample = float(os.getenv('SAMPLE_PCT', '20'))
    s = '' if sample >= 100 else f' TABLESAMPLE SYSTEM ({sample:g})'
    q = f"""SELECT PESO_BRUTO_TN - TARA_TN AS PESO_NETO, HUMEDAD_PCT, IMPUREZAS_PCT,
        ACEITE_PCT, PROTEINA_PCT, TEMPERATURA_GRANO_C,
        EXTRACT(MONTH FROM FECHA_HORA) AS MES, ID_SILO, CALIDAD_GRANO
        FROM ARQUI1.CAPA_BRONCE.RECEPCION_GRANO{s}
        WHERE PESO_BRUTO_TN - TARA_TN BETWEEN 1 AND 200
          AND FECHA_HORA BETWEEN '2020-01-01' AND '2027-01-01' AND CALIDAD_GRANO IS NOT NULL"""
    try:
        from snowflake.snowpark.context import get_active_session
    except ImportError:
        return _cargar_local(q, sample)
    session = get_active_session()
    print('Origen: sesion Snowflake (Snowpark), sin credenciales')
    return session.sql(q).to_pandas(), session
def _cargar_local(q, sample):
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
    cur = c.cursor(); cur.execute(q)
    import pandas as pd
    print(f'Origen: connector local (muestra {sample:g}%)')
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
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 confusion_matrix, roc_auc_score, classification_report)
    df, session = cargar()
    df.columns = df.columns.str.lower()
    num = [x.lower() for x in NUM]
    cat = [x.lower() for x in CAT]
    for col in num:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['calidad_grano']).copy()
    print(f'Filas: {len(df)}')
    print(df['calidad_grano'].value_counts().to_string())
    X = df[num + cat]
    y = df['calidad_grano'].astype(str)
    Xtr, Xev, ytr, yev = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    Xva, Xte, yva, yte = train_test_split(Xev, yev, test_size=0.50, stratify=yev, random_state=SEED)
    print(f'train={len(Xtr)} val={len(Xva)} test={len(Xte)}')
    pre = ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), num),
        ('cat', make_pipeline(SimpleImputer(strategy='constant', fill_value=-1),
                               OneHotEncoder(handle_unknown='ignore')), cat)])
    cand = {
        'LogReg': make_pipeline(pre, LogisticRegression(max_iter=500, class_weight='balanced', random_state=SEED)),
        'HGB': HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=SEED),
        'PCA+LogReg': make_pipeline(pre, PCA(n_components=0.95, svd_solver='full', random_state=SEED),
                                     LogisticRegression(max_iter=500, class_weight='balanced', random_state=SEED)),
    }
    print('=== VALIDATION (F1-macro) ===')
    best, bn = -1, ''
    from sklearn.preprocessing import label_binarize
    order = sorted(yva.unique())
    for n, m in cand.items():
        if n == 'HGB':
            Xh = Xtr.copy()
            for col in num:
                Xh[col] = pd.to_numeric(Xh[col], errors='coerce')
            Xh[cat] = Xh[cat].fillna(-1).astype(int)
            Xvh = Xva.copy()
            for col in num:
                Xvh[col] = pd.to_numeric(Xvh[col], errors='coerce')
            Xvh[cat] = Xvh[cat].fillna(-1).astype(int)
            m.fit(Xh, ytr)
            p, s = m.predict(Xvh), m.predict_proba(Xvh)
        else:
            m.fit(Xtr, ytr)
            p, s = m.predict(Xva), m.predict_proba(Xva)
        f1 = f1_score(yva, p, average='macro')
        auc = f' AUC={roc_auc_score(label_binarize(yva, classes=order), s, multi_class="ovr"):.3f}'
        print(f'{n}: F1macro={f1:.3f}{auc}')
        if f1 > best:
            best, bn = f1, n
    print(f'Campeon: {bn}')
    print('=== REPORTE (TEST unico) ===')
    m = cand[bn]
    if bn == 'HGB':
        Xh = Xte.copy()
        for col in num:
            Xh[col] = pd.to_numeric(Xh[col], errors='coerce')
        Xh[cat] = Xh[cat].fillna(-1).astype(int)
        p = m.predict(Xh)
    else:
        p = m.predict(Xte)
    print(f'Modelo={bn} acc={accuracy_score(yte, p):.3f} bal={balanced_accuracy_score(yte, p):.3f} F1macro={f1_score(yte, p, average="macro"):.3f}')
    print(classification_report(yte, p, digits=3))
    print('Matriz (filas=real, cols=pred):')
    print(pd.DataFrame(confusion_matrix(yte, p, labels=sorted(yte.unique())), index=sorted(yte.unique()), columns=sorted(yte.unique())).to_string())
if __name__ == '__main__':
    main()
