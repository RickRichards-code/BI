#!/usr/bin/env python3
"""Mineria pura (sin notebook): segmentacion de productores k-means + DBSCAN.
Uso:  python mineria_productores.py
Lee data/productores_features_v2.csv (o Snowflake si hay SNOWFLAKE_PASSWORD).
Imprime el reporte para evaluacion y guarda data/productores_segmentos_v2.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)
SEED = 42
FEATS = ['n_recepciones', 'volumen_total_tn', 'ticket_medio_tn', 'merma_total_tn',
         'humedad_media', 'humedad_std', 'humedad_max', 'pct_hum_crit',
         'impurezas_media', 'impurezas_std', 'impurezas_max', 'pct_imp_crit',
         'proteina_media', 'aceite_media', 'pct_descuento', 'max_merma_evento']
def cargar():
    try:
        import snowflake.connector
        c = snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USER', 'ENRRIQUE'), password=os.environ['SNOWFLAKE_PASSWORD'],
            account=os.getenv('SNOWFLAKE_ACCOUNT', 'AVBVHGL-WZ57062'),
            database='AIRBYTE_DATABASE', warehouse='COMPUTE_WH')
        q = open('sql/productores_features.sql', encoding='utf-8').read()
        cur = c.cursor(); cur.execute(q)
        df = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
        print('Origen: Snowflake en vivo'); return df
    except Exception as e:
        print(f'Snowflake no disponible ({type(e).__name__}): uso CSV local')
        return pd.read_csv('data/productores_features_v2.csv')
def main():
    df = cargar()
    df.columns = df.columns.str.lower()
    d = df[(df['id_productor'] != 525)].dropna(subset=FEATS).copy()
    for c in FEATS:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna(subset=FEATS)
    print(f'Productores utiles (sin 525): {len(d)}')
    Xs = StandardScaler().fit_transform(np.log1p(d[FEATS].clip(lower=0)))
    print('=== LEADERBOARD k-means ===')
    res = {}
    for k in range(2, 8):
        lab = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(Xs)
        res[k] = (silhouette_score(Xs, lab), davies_bouldin_score(Xs, lab))
        print(f'K={k} sil={res[k][0]:.3f} DB={res[k][1]:.3f} '
              f'sizes={pd.Series(lab).value_counts().sort_index().to_dict()}')
    K = max(res, key=lambda k: res[k][0])
    final = KMeans(n_clusters=K, n_init=50, random_state=SEED).fit_predict(Xs)
    ari = adjusted_rand_score(
        KMeans(n_clusters=K, n_init=20, random_state=1).fit_predict(Xs),
        KMeans(n_clusters=K, n_init=20, random_state=2).fit_predict(Xs))
    print(f'=== K={K} sil={silhouette_score(Xs, final):.3f} ARI={ari:.3f} ===')
    d['segmento'] = final
    print(d.groupby('segmento')[FEATS].mean().round(2).to_string())
    # DBSCAN atipicos
    nn = 2 * len(FEATS)
    from sklearn.neighbors import NearestNeighbors
    dist, _ = NearestNeighbors(n_neighbors=nn).fit(Xs).kneighbors(Xs)
    eps = float(np.quantile(np.sort(dist[:, -1]), 0.95))
    db = DBSCAN(eps=eps, min_samples=nn).fit(Xs)
    print(f'DBSCAN eps={eps:.3f}: {(db.labels_ == -1).sum()} atipicos ({(db.labels_ == -1).mean():.1%})')
    out = d[['id_productor', 'nombre_completo', 'tipo_productor', 'municipio', 'segmento']].copy()
    out['atipico_dbscan'] = (db.labels_ == -1).astype(int)
    out.to_csv('data/productores_segmentos_v2.csv', index=False, encoding='utf-8')
    print('Guardado data/productores_segmentos_v2.csv')
if __name__ == '__main__':
    main()
