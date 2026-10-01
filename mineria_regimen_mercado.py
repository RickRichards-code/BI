#!/usr/bin/env python3
"""Regimenes del mercado sojero + causalidad climatica (Kaggle real en Bronce).
k-means (regimenes) + DBSCAN (crisis) + PCA + ADF + Granger (NINO34 -> SOYA).
IMPORTANCIA: operar/cubrir segun regimen y anticipar precio con El Nino.
Uso:  python mineria_regimen_mercado.py
Requiere SNOWFLAKE_PASSWORD. Opcional: statsmodels (pip install statsmodels).
Guarda data/mercado_regimen_mensual.csv
"""
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
SEED = 42
LVL = ['SOYA_GRANO', 'SOYA_ACEITE', 'SOYA_HARINA', 'MAIZ', 'PETROLEO', 'UREA', 'DAP', 'POTASIO', 'NINO34_ANOMALIA_C']
RET = ['DLOG_SOYA_GRANO', 'DLOG_SOYA_ACEITE', 'DLOG_SOYA_HARINA', 'DLOG_PETROLEO', 'DLOG_MAIZ']
def conectar():
    import snowflake.connector
    pwd = os.getenv('SNOWFLAKE_PASSWORD', '').strip().strip('\'"')
    if not pwd:
        raise SystemExit("Sin SNOWFLAKE_PASSWORD: export SNOWFLAKE_PASSWORD='clave'")
    kw = dict(user=os.getenv('SNOWFLAKE_USER', 'ENRRIQUE'), password=pwd,
              account=os.getenv('SNOWFLAKE_ACCOUNT', 'AVBVHGL-WZ57062'),
              database='AIRBYTE_DATABASE', warehouse='COMPUTE_WH', login_timeout=60)
    if os.getenv('SNOWFLAKE_ROLE'):
        kw['role'] = os.getenv('SNOWFLAKE_ROLE')
    return snowflake.connector.connect(**kw)
def main():
    c = conectar(); cur = c.cursor()
    cur.execute('SELECT * FROM AIRBYTE_DATABASE.BRONCE_ECONOMETRIA.PRECIOS_ENSO_MENSUAL ORDER BY MES')
    df = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
    df.columns = df.columns.str.lower()
    df['mes'] = pd.to_datetime(df['mes'], errors='coerce')
    for col in LVL + RET:
        if col.lower() in df.columns:
            df[col.lower()] = pd.to_numeric(df[col.lower()], errors='coerce')
    feats = [x.lower() for x in RET + ['nino34_anomalia_c']]
    d = df.dropna(subset=feats).copy()
    print(f'Meses utiles: {len(d)} ({d["mes"].min():%Y-%m} -> {d["mes"].max():%Y-%m})')
    Xs = StandardScaler().fit_transform(d[feats])
    # 1. Estacionariedad ADF (estadistica)
    try:
        from statsmodels.tsa.stattools import adfuller
        print('ADF p-valores (H0: no estacionaria):')
        for col in feats:
            print(f'  {col}: p={adfuller(d[col].dropna())[1]:.4f}')
    except ImportError:
        print('Sin statsmodels: salto ADF/Granger (pip install statsmodels)')
        return
    # 2. Regimenes k-means
    res = {}
    for k in range(2, 7):
        lab = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(Xs)
        res[k] = silhouette_score(Xs, lab)
        print(f'K={k} sil={res[k]:.3f} sizes={pd.Series(lab).value_counts().sort_index().to_dict()}')
    K = max(res, key=res.get)
    d['regimen'] = KMeans(n_clusters=K, n_init=50, random_state=SEED).fit_predict(Xs)
    print(f'Regimen K={K}')
    print(d.groupby('regimen')[feats].mean().round(4).to_string())
    # ANOVA por regimen (que variables separan)
    from scipy.stats import f_oneway
    print('ANOVA F por variable (separacion entre regimenes):')
    for col in feats:
        grupos = [g[col].values for _, g in d.groupby('regimen')]
        print(f'  {col}: F={f_oneway(*grupos).statistic:.1f} p={f_oneway(*grupos).pvalue:.2e}')
    # 3. DBSCAN crisis (meses raros)
    db = DBSCAN(eps=1.2, min_samples=12).fit(Xs)
    d['crisis'] = (db.labels_ == -1).astype(int)
    print(f'DBSCAN meses crisis: {int(d["crisis"].sum())}')
    print(d.loc[d['crisis'] == 1, ['mes']].to_string())
    print('(detalle numerico en data/mercado_regimen_mensual.csv)')
    # 4. Granger: Nino -> soya (estadistica estrella)
    from statsmodels.tsa.stattools import grangercausalitytests
    g = d[['dlog_soya_grano', 'nino34_anomalia_c']].dropna()
    print('Granger NINO34 -> SOYA (p<0.05 = causa):')
    import io as _io
    import contextlib as _ctx
    _buf = _io.StringIO()
    with _ctx.redirect_stdout(_buf):
        out = grangercausalitytests(g[['dlog_soya_grano', 'nino34_anomalia_c']], maxlag=6)
    for lag in range(1, 7):
        p = out[lag][0]['ssr_ftest'][1]
        print(f'  lag {lag}: p={p:.4f} {"CAUSA" if p < 0.05 else ""}')
    # 5. PCA (para biplot en notebook/defensa)
    pc = PCA(n_components=2, random_state=SEED).fit_transform(Xs)
    print(f'PCA var explicada: {PCA(n_components=2, random_state=SEED).fit(Xs).explained_variance_ratio_.round(3)}')
    d[['mes']].assign(regimen=d['regimen'], crisis=d['crisis']).to_csv('data/mercado_regimen_mensual.csv', index=False, encoding='utf-8')
    print('Guardado data/mercado_regimen_mensual.csv')
if __name__ == '__main__':
    main()
