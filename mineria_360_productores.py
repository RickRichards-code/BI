#!/usr/bin/env python3
"""Mineria 360 productores: cumplimiento, estacionalidad, lealtad, drivers de merma (GBR) y basis.
Uso: python mineria_360_productores.py  (Snowflake en vivo)
Guarda data/segmentos_360.csv + imprime hallazgos.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
SEED = 42
def conectar():
    import snowflake.connector
    pwd = os.getenv('SNOWFLAKE_PASSWORD', '').strip().strip('\'"')
    if not pwd:
        raise SystemExit("Sin SNOWFLAKE_PASSWORD")
    kw = dict(user=os.getenv('SNOWFLAKE_USER', 'ENRRIQUE'), password=pwd,
              account=os.getenv('SNOWFLAKE_ACCOUNT', 'AVBVHGL-WZ57062'),
              database='AIRBYTE_DATABASE', warehouse='COMPUTE_WH', login_timeout=60)
    if os.getenv('SNOWFLAKE_ROLE'):
        kw['role'] = os.getenv('SNOWFLAKE_ROLE')
    return snowflake.connector.connect(**kw)
def leer(q):
    cur = conectar().cursor(); cur.execute(q)
    df = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
    df.columns = df.columns.str.lower()
    return df
def main():
    print('== 1. Compliance + comportamiento por productor ==')
    p = leer("""SELECT p.ID_PRODUCTOR, p.TIPO_PRODUCTOR,
        COUNT(*) AS N, SUM(r.PESO_BRUTO_TN - r.TARA_TN) AS VOL,
        AVG(CASE WHEN r.HUMEDAD_PCT > 14 THEN 1.0 ELSE 0.0 END) AS P_CRIT,
        STDDEV(r.HUMEDAD_PCT) AS HUM_STD, MAX(r.HUMEDAD_PCT) AS HUM_MAX,
        COUNT(DISTINCT r.ID_SILO) AS N_SILOS,
        COUNT(DISTINCT TO_CHAR(r.FECHA_HORA, 'YYYY-MM')) AS N_MESES,
        MAX(TO_CHAR(r.FECHA_HORA, 'YYYY-MM')) AS ULTIMO_MES,
        SUM(CASE WHEN EXTRACT(HOUR FROM r.FECHA_HORA) BETWEEN 22 AND 23
                  OR EXTRACT(HOUR FROM r.FECHA_HORA) BETWEEN 0 AND 5 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS P_NOCT,
        SUM((r.PESO_BRUTO_TN - r.TARA_TN) * (r.PESO_BRUTO_TN - r.TARA_TN)) / NULLIF(SUM(r.PESO_BRUTO_TN - r.TARA_TN), 0) AS TICKET_POND
        FROM AIRBYTE_DATABASE.BRONCE.RECEPCION_GRANO r
        LEFT JOIN AIRBYTE_DATABASE.BRONCE.CONTRATO_COMPRA c ON r.ID_CONTRATO = c.ID_CONTRATO
        JOIN AIRBYTE_DATABASE.BRONCE.PRODUCTOR p ON c.ID_PRODUCTOR = p.ID_PRODUCTOR
        WHERE r.PESO_BRUTO_TN - r.TARA_TN BETWEEN 1 AND 200
          AND r.FECHA_HORA BETWEEN '2020-01-01' AND '2027-01-01'
        GROUP BY 1, 2""")
    cc = leer("""SELECT ID_PRODUCTOR, SUM(VOLUMEN_COMPROMETIDO_TN) AS COMP
                 FROM (SELECT DISTINCT ID_CONTRATO, ID_PRODUCTOR, VOLUMEN_COMPROMETIDO_TN
                       FROM AIRBYTE_DATABASE.BRONCE.CONTRATO_COMPRA) GROUP BY 1""")
    cc['comp'] = pd.to_numeric(cc['comp'], errors='coerce')
    p = p.merge(cc[['id_productor', 'comp']], on='id_productor', how='left')
    for col in ['n', 'vol', 'comp', 'p_crit', 'hum_std', 'hum_max', 'n_silos', 'n_meses', 'p_noct', 'ticket_pond']:
        p[col] = pd.to_numeric(p[col], errors='coerce')
    p['cumplimiento'] = p['vol'] / p['comp'].replace(0, np.nan)
    p['pct_cubierto'] = (p['comp'] / p['vol'].replace(0, np.nan)).clip(0, 1)
    print(f'Productores: {len(p)} | cumplimiento mediano: {p["cumplimiento"].median():.2f} | cubierto mediano: {p["pct_cubierto"].median():.2%}')
    print('Top incumplidos (<50%, vol>1000):')
    print(p[(p['cumplimiento'] < 0.5) & (p['vol'] > 1000)][['id_productor', 'tipo_productor', 'vol', 'comp', 'cumplimiento']].head(10).to_string())
    print('Fantasmas (>200% entregado):')
    print(p[p['cumplimiento'] > 2][['id_productor', 'tipo_productor', 'vol', 'comp', 'cumplimiento']].head(10).to_string())
    # Segmentacion 360
    F = ['n', 'vol', 'cumplimiento', 'p_crit', 'hum_std', 'n_silos', 'n_meses', 'p_noct']
    d = p.dropna(subset=F).copy()
    Xs = StandardScaler().fit_transform(np.log1p(d[F].clip(lower=0)))
    for k in (3, 4, 5):
        lab = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(Xs)
        print(f'K={k} sil={silhouette_score(Xs, lab):.3f} sizes={pd.Series(lab).value_counts().sort_index().to_dict()}')
    d['seg360'] = KMeans(n_clusters=4, n_init=20, random_state=SEED).fit_predict(Xs)
    print(d.groupby('seg360')[F + ['cumplimiento']].mean().round(2).to_string())
    d[['id_productor', 'seg360']].to_csv('data/segmentos_360.csv', index=False, encoding='utf-8')
    print('== 2. Drivers de merma (GBR 200k muestra) ==')
    m = leer("""SELECT (PESO_BRUTO_TN - TARA_TN) AS PESO_NETO, HUMEDAD_PCT, IMPUREZAS_PCT, ACEITE_PCT,
        PROTEINA_PCT, TEMPERATURA_GRANO_C, EXTRACT(MONTH FROM FECHA_HORA) AS MES,
        EXTRACT(HOUR FROM FECHA_HORA) AS HORA, ID_SILO
        FROM AIRBYTE_DATABASE.BRONCE.RECEPCION_GRANO TABLESAMPLE SYSTEM (15)
        WHERE PESO_BRUTO_TN - TARA_TN BETWEEN 1 AND 200 AND HUMEDAD_PCT IS NOT NULL""")
    for col in m.columns:
        if col != 'id_silo':
            m[col] = pd.to_numeric(m[col], errors='coerce')
    m = m.dropna().copy()
    m['merma'] = m['peso_neto'] * (np.clip(m['humedad_pct'] / 100 - 0.14, 0, None) + np.clip(m['impurezas_pct'] / 100 - 0.01, 0, None))
    FX = ['peso_neto', 'humedad_pct', 'impurezas_pct', 'aceite_pct', 'proteina_pct', 'temperatura_grano_c', 'mes', 'hora']
    g = HistGradientBoostingRegressor(max_iter=200, random_state=SEED).fit(m[FX], m['merma'])
    print(f'R2 train={g.score(m[FX], m["merma"]):.3f} n={len(m)}')
    imp = permutation_importance(g, m[FX], m['merma'], n_repeats=3, random_state=SEED)
    print(pd.Series(imp.importances_mean, index=FX).sort_values(ascending=False).round(4).to_string())
    print('== 3. Basis por segmento de cliente ==')
    v = leer("""SELECT C.SEGMENTO, V.PRIMA_BASIS_USD FROM AIRBYTE_DATABASE.BRONCE.VENTA V
        LEFT JOIN AIRBYTE_DATABASE.BRONCE.CONTRATO_VENTA CV ON V.ID_CONTRATO_VENTA = CV.ID_CONTRATO_VENTA
        LEFT JOIN AIRBYTE_DATABASE.BRONCE.CLIENTE C ON CV.ID_CLIENTE = C.ID_CLIENTE""")
    v['prima_basis_usd'] = pd.to_numeric(v['prima_basis_usd'], errors='coerce')
    print(v.groupby('segmento')['prima_basis_usd'].agg(['mean', 'std', 'count']).round(2).to_string())
if __name__ == '__main__':
    main()
