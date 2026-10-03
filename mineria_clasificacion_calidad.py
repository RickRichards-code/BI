#!/usr/bin/env python3
"""Clasificacion CALIDAD de recepcion (Bronce cuenta nueva): ESTANDAR vs CONDICIONADO.
Diseno honesto para clases 99.2/0.8%: split por columna SPLIT (train/val/test),
seleccion en validation, evaluacion FINAL unica en test. Baselines + 2 modelos.
Uso:  python mineria_clasificacion_calidad.py   (requiere SNOWFLAKE_* en entorno)
Imprime REPORTE PARA EVALUACION para pegar y revisar.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)
SEED = 42
NUM = ['MES_COS', 'MES_SEN', 'TEMP_C_7D', 'LLUVIA_MM_7D', 'SUPERFICIE_HA',
       'DIAS_DESDE_CONTRATO', 'OBJETIVO_HUMEDAD_PCT', 'PRECIO_ACORDADO_USD_TN',
       'HUMEDAD_RELATIVA_PCT_7D', 'VOLUMEN_COMPROMETIDO_TN']
CAT = ['TIPO_PRODUCTOR']
def conectar():
    import snowflake.connector
    pwd = os.getenv('SNOWFLAKE_PASSWORD', '').strip().strip('\'"')
    if not pwd:
        try:
            import pathlib as _pl
            pwd = _pl.Path.home().joinpath('.snowflake_pwd').read_text(encoding='utf-8').strip()
        except Exception:
            pwd = ''
    if not pwd:
        raise SystemExit("Sin password: export SNOWFLAKE_PASSWORD='clave' o ~/.snowflake_pwd")
    kw = dict(user=os.getenv('SNOWFLAKE_USER', 'ENRIQUE'), password=pwd,
              account=os.getenv('SNOWFLAKE_ACCOUNT', 'WMNAMCT-LN60692'),
              database=os.getenv('SNOWFLAKE_DB', 'ARQUI1'),
              schema=os.getenv('SNOWFLAKE_SCHEMA', 'CAPA_BRONCE'),
              warehouse=os.getenv('SNOWFLAKE_WH', 'COMPUTE_WH'), login_timeout=60)
    if os.getenv('SNOWFLAKE_ROLE'):
        kw['role'] = os.getenv('SNOWFLAKE_ROLE')
    return snowflake.connector.connect(**kw)
def main():
    c = conectar(); cur = c.cursor()
    cols = [x for x in NUM + CAT + ['SPLIT', 'OBJETIVO_CALIDAD']]
    cur.execute(f"SELECT {', '.join(cols)} FROM CALIDAD_RECEPCIONES")
    df = pd.DataFrame(cur.fetchall(), columns=[d[0].lower() for d in cur.description])
    num = [x.lower() for x in NUM]
    cat = [x.lower() for x in CAT]
    print(f'Filas: {len(df)}')
    print('Distribucion:\n', pd.crosstab(df['objetivo_calidad'], df['split']).to_string())
    print('Nulos por columna:\n', df.isna().sum().to_string())
    df['y'] = (df['objetivo_calidad'] == 'CONDICIONADO').astype(int)
    pre = ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), num),
        ('cat', make_pipeline(SimpleImputer(strategy='constant', fill_value='FALTA'),
                               OneHotEncoder(handle_unknown='ignore')), cat)])
    tr, va, te = df[df['split'] == 'train'], df[df['split'] == 'validation'], df[df['split'] == 'test']
    Xtr, ytr, Xva, yva, Xte, yte = tr[num + cat], tr['y'], va[num + cat], va['y'], te[num + cat], te['y']
    print(f'train={len(tr)} (pos={int(ytr.sum())}) val={len(va)} (pos={int(yva.sum())}) test={len(te)} (pos={int(yte.sum())})')
    modelos = {
        'Dummy': DummyClassifier(strategy='stratified', random_state=SEED),
        'Logistica': make_pipeline(pre, LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)),
        'RandomForest': make_pipeline(pre, RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight='balanced_subsample', n_jobs=-1, random_state=SEED)),
    }
    print('=== VALIDATION (seleccion) ===')
    best, bn = -1, ''
    for n, m in modelos.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xva)
        s = m.predict_proba(Xva)[:, 1]
        f1 = f1_score(yva, p, zero_division=0)
        print(f'{n}: acc={accuracy_score(yva, p):.3f} bal={balanced_accuracy_score(yva, p):.3f} '
              f'prec={precision_score(yva, p, zero_division=0):.3f} rec={recall_score(yva, p):.3f} '
              f'F1={f1:.3f} AUC={roc_auc_score(yva, s):.3f} AP={average_precision_score(yva, s):.3f}')
        if n != 'Dummy' and f1 > best:
            best, bn = f1, n
    print(f'Campeon: {bn}')
    print('=== REPORTE PARA EVALUACION (TEST unico) ===')
    m = modelos[bn]
    p = m.predict(Xte)
    s = m.predict_proba(Xte)[:, 1]
    print(f'Modelo={bn} Filas test={len(te)}')
    print(f"accuracy={accuracy_score(yte, p):.3f} bal_acc={balanced_accuracy_score(yte, p):.3f} "
          f"precision={precision_score(yte, p, zero_division=0):.3f} recall={recall_score(yte, p):.3f} "
          f"F1={f1_score(yte, p, zero_division=0):.3f} AUC={roc_auc_score(yte, s):.3f} AP={average_precision_score(yte, s):.3f}")
    print('Matriz test [[TN FP][FN TP]]:')
    print(confusion_matrix(yte, p).tolist())
if __name__ == '__main__':
    main()
