#!/usr/bin/env python3
"""Mineria en datasets solidos: regimenes operativos IoT + regimenes de mercado CBOT.
k-means / MiniBatchKMeans (segmentos) + DBSCAN (atipicos). IMPORTANCIA:
  IoT -> mantenimiento preventivo por silo/sensor (que vigilar y cuando ventilar).
  CBOT -> timing de cobertura (en que regimen operar / no operar).
Uso:  python mineria_regimenes.py                 (100%)
      SAMPLE_PCT=5 python mineria_regimenes.py      (prueba)
Requiere SNOWFLAKE_PASSWORD. Guarda perfiles en data/.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.metrics import silhouette_score
SEED = 42
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
def leer(query, sample):
    c = conectar(); cur = c.cursor(); cur.execute(query)
    df = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
    df.columns = df.columns.str.lower()
    return df
def mineria_iot(sample):
    print('=== IoT: regimenes operativos ===')
    s = '' if float(sample) >= 100 else f' TABLESAMPLE SYSTEM ({float(sample)})'
    df = leer(f"""SELECT SENSOR_ID, TIMESTAMP_LECTURA, TEMPERATURA_C, HUMEDAD_RELATIVA_PCT,
                         PRECIPITACION_MM, VOLTAJE_BATERIA, CALIDAD_SENAL_DBM
                  FROM AIRBYTE_DATABASE.BRONCE_CLIMATICO.IOT_TELEMETRIA_CRUDA{s}""", sample)
    for col in ['temperatura_c', 'humedad_relativa_pct', 'precipitacion_mm', 'voltaje_bateria', 'calidad_senal_dbm']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['ts'] = pd.to_datetime(df['timestamp_lectura'], errors='coerce')
    df['hora'] = df['ts'].dt.hour + df['ts'].dt.minute / 60
    feats = ['temperatura_c', 'humedad_relativa_pct', 'precipitacion_mm', 'calidad_senal_dbm', 'hora']
    d = df.dropna(subset=feats).copy()
    print(f'Lecturas utiles: {len(d)} (voltaje constante -> fuera)')
    Xs = StandardScaler().fit_transform(d[feats])
    best, bestk = -1, 3
    for k in (3, 4, 5):
        lab = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init=5, random_state=SEED).fit_predict(Xs)
        sil = silhouette_score(Xs[np.random.RandomState(SEED).choice(len(Xs), min(50000, len(Xs)), replace=False)], lab[np.random.RandomState(SEED).choice(len(Xs), min(50000, len(Xs)), replace=False)]) if len(Xs) > 50000 else silhouette_score(Xs, lab)
        print(f'K={k} sil={sil:.3f} sizes={pd.Series(lab).value_counts().sort_index().to_dict()}')
        if sil > best:
            best, bestk = sil, k
    km = MiniBatchKMeans(n_clusters=bestk, batch_size=4096, n_init=10, random_state=SEED).fit(Xs)
    d['regimen'] = km.labels_
    print(d.groupby('regimen')[feats].mean().round(2).to_string())
    prof = d.groupby('sensor_id')['regimen'].agg(lambda x: x.mode().iloc[0])
    crit = d[d['regimen'] == d.groupby('regimen')['temperatura_c'].mean().idxmax()]['sensor_id'].value_counts().head(5)
    print('Sensores mas tiempo en regimen caliente:'); print(crit.to_string())
    # DBSCAN atipicos sobre muestra (salud de red)
    m = min(100000, len(Xs))
    idx = np.random.RandomState(SEED).choice(len(Xs), m, replace=False)
    db = DBSCAN(eps=0.9, min_samples=50).fit(Xs[idx])
    print(f'DBSCAN (muestra {m}): {(db.labels_ == -1).sum()} atipicos ({(db.labels_ == -1).mean():.2%})')
    prof.to_csv('data/iot_sensor_regimen.csv', encoding='utf-8')
    print('Guardado data/iot_sensor_regimen.csv')
def mineria_mercado():
    print('=== CBOT: regimenes intradiarios ===')
    df = leer("""SELECT TICKER, TIMESTAMP_MERCADO, PRECIO_COTIZACION, VOLUMEN_TICK
                 FROM AIRBYTE_DATABASE.BRONCEFINANCIERO.COTIZACIONES_INTRADIARIAS""", 100)
    df['precio_cotizacion'] = pd.to_numeric(df['precio_cotizacion'], errors='coerce')
    df['volumen_tick'] = pd.to_numeric(df['volumen_tick'], errors='coerce')
    df['ts'] = pd.to_datetime(df['timestamp_mercado'], errors='coerce')
    df['hora'] = df['ts'].dt.floor('h')
    df = df.sort_values(['ticker', 'ts'])
    df['ret'] = df.groupby('ticker')['precio_cotizacion'].pct_change()
    h = df.groupby(['ticker', 'hora']).agg(volatilidad=('ret', lambda s: s.std()),
        volumen=('volumen_tick', 'sum'), rango=('precio_cotizacion', lambda s: s.max() - s.min()),
        n_ticks=('precio_cotizacion', 'count')).dropna()
    print(f'Horas-sesion: {len(h)}')
    Xs = StandardScaler().fit_transform(np.log1p(h[['volatilidad', 'volumen', 'rango']].clip(lower=0)))
    best, bestk = -1, 2
    for k in (2, 3, 4):
        lab = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(Xs)
        sil = silhouette_score(Xs, lab)
        print(f'K={k} sil={sil:.3f} sizes={pd.Series(lab).value_counts().sort_index().to_dict()}')
        if sil > best:
            best, bestk, bestlab = sil, k, lab
    h['regimen'] = bestlab
    print(f'Regimen elegido K={bestk}')
    print(h.groupby('regimen')[['volatilidad', 'volumen', 'rango', 'n_ticks']].mean().round(4).to_string())
    h.to_csv('data/mercado_regimen_hora.csv', encoding='utf-8')
    print('Guardado data/mercado_regimen_hora.csv')
if __name__ == '__main__':
    sample = float(os.getenv('SAMPLE_PCT', '100'))
    os.makedirs('data', exist_ok=True)
    mineria_iot(sample)
    mineria_mercado()
