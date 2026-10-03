#!/usr/bin/env python3
"""Predictivo con reduccion de dimensionalidad (Bronce cuenta nueva).
CALIDAD_RECEPCIONES -> ESTANDAR vs CONDICIONADO. Compara: sin reduccion vs
PCA(95% var) vs LDA(1 eje supervisado), con LogReg y RF. PCA/LDA ajustados SOLO
en train (sin leakage). Seleccion en validation, test final unico.
Uso:  python mineria_clasificacion_reduccion.py
Guarda docs/img/reduccion.png + imprime REPORTE PARA EVALUACION.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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
    df['y'] = (df['objetivo_calidad'] == 'CONDICIONADO').astype(int)
    pre = ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), num),
        ('cat', make_pipeline(SimpleImputer(strategy='constant', fill_value='FALTA'),
                               OneHotEncoder(handle_unknown='ignore')), cat)])
    tr = df[df['split'] == 'train']; va = df[df['split'] == 'validation']; te = df[df['split'] == 'test']
    Xtr, ytr = tr[num + cat], tr['y']
    Xva, yva = va[num + cat], va['y']
    Xte, yte = te[num + cat], te['y']
    print(f'train={len(tr)}(+{int(ytr.sum())}) val={len(va)}(+{int(yva.sum())}) test={len(te)}(+{int(yte.sum())})')
    Xtr_p = pre.fit_transform(Xtr); Xva_p = pre.transform(Xva); Xte_p = pre.transform(Xte)
    print(f'dim original: {Xtr_p.shape[1]}')
    pca = PCA(n_components=0.95, random_state=SEED).fit(Xtr_p)
    print(f'PCA 95% var -> {pca.n_components_} ejes')
    lda = LDA().fit(Xtr_p, ytr)
    print('LDA -> 1 eje supervisado (binario)')
    def _pre():
        return ColumnTransformer([
            ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), num),
            ('cat', make_pipeline(SimpleImputer(strategy='constant', fill_value='FALTA'),
                                   OneHotEncoder(handle_unknown='ignore')), cat)])
    lr = dict(max_iter=1000, class_weight='balanced', random_state=SEED)
    rf = dict(n_estimators=300, min_samples_leaf=5, class_weight='balanced_subsample', n_jobs=-1, random_state=SEED)
    cand = {
        'RF_base': make_pipeline(_pre(), RandomForestClassifier(**rf)),
        'PCA+LogReg': make_pipeline(_pre(), PCA(n_components=0.95, random_state=SEED), LogisticRegression(**lr)),
        'PCA+RF': make_pipeline(_pre(), PCA(n_components=0.95, random_state=SEED), RandomForestClassifier(**rf)),
        'LDA+LogReg': make_pipeline(_pre(), LDA(), LogisticRegression(**lr)),
    }
    print('=== VALIDATION ===')
    best, bn = -1, ''
    for n, m in cand.items():
        m.fit(Xtr, ytr)
        p, s = m.predict(Xva), m.predict_proba(Xva)[:, 1]
        f1 = f1_score(yva, p, zero_division=0)
        print(f'{n}: F1={f1:.3f} AUC={roc_auc_score(yva, s):.3f} AP={average_precision_score(yva, s):.3f} rec={recall_score(yva, p):.3f}')
        if f1 > best:
            best, bn = f1, n
    print(f'Campeon: {bn}')
    print('=== REPORTE PARA EVALUACION (TEST) ===')
    m = cand[bn]
    p, s = m.predict(Xte), m.predict_proba(Xte)[:, 1]
    print(f'Modelo={bn} F1={f1_score(yte, p, zero_division=0):.3f} AUC={roc_auc_score(yte, s):.3f} '
          f'AP={average_precision_score(yte, s):.3f} rec={recall_score(yte, p):.3f} prec={precision_score(yte, p, zero_division=0):.3f}')
    print('Matriz [[TN FP][FN TP]]:', confusion_matrix(yte, p).tolist())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    pv = PCA(n_components=2, random_state=SEED).fit(Xtr_p).transform(Xte_p)
    ax[0].scatter(pv[yte.values == 0, 0], pv[yte.values == 0, 1], s=8, alpha=.4, label='ESTANDAR')
    ax[0].scatter(pv[yte.values == 1, 0], pv[yte.values == 1, 1], s=30, alpha=.9, label='CONDICIONADO')
    ax[0].legend(); ax[0].set_title('PCA-2D test')
    ax[1].plot(np.cumsum(PCA(random_state=SEED).fit(Xtr_p).explained_variance_ratio_), 'o-')
    ax[1].set_title('Varianza acumulada PCA'); ax[1].set_xlabel('ejes')
    lz = LDA().fit(Xtr_p, ytr).transform(Xte_p).ravel()
    ax[2].hist(lz[yte.values == 0], bins=30, alpha=.5, label='ESTANDAR')
    ax[2].hist(lz[yte.values == 1], bins=10, alpha=.8, label='CONDICIONADO')
    ax[2].legend(); ax[2].set_title('Proyeccion LDA (1 eje)')
    plt.tight_layout()
    os.makedirs('docs/img', exist_ok=True)
    plt.savefig('docs/img/reduccion.png', dpi=110)
    print('docs/img/reduccion.png OK')
if __name__ == '__main__':
    main()
