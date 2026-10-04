#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ML clasificacion en Snowflake: riesgo termico en silos (temp > 30C).
Replica del notebook v3 (recall ~90% con umbral operativo 0.32), reconstruido
desde Bronce cuenta nueva (ARQUI1.CAPA_BRONCE.IOT_TELEMETRIA_CRUDA) con lags
por sensor. Snowpark-first (sin credenciales en Snowflake), SAMPLE_PCT=30.
UMBRAL_OP=0.32, ESCRIBIR=1 guarda PRED_RIESGO_IOT. PNGs en docs/img/riesgo_iot.png
"""
import os
SEED = 42
UMBRAL_OP = float(os.getenv('UMBRAL_OP', '0.32'))
LEAD_N = int(os.getenv('LEAD_N', '1'))  # 0 = ahora (diagnostico), >=1 = alerta anticipada
FEATS = ['humedad_relativa_pct', 'precipitacion_mm', 'calidad_senal_dbm',
         'hora_dia', 'hora_sin', 'hora_cos', 'mes', 'hum_lag_1',
         'temp_media_6h', 'temp_std_6h', 'temp_max_6h', 'hum_media_6h',
         'delta_3h', 'humidex']
def cargar():
    sample = float(os.getenv('SAMPLE_PCT', '30'))
    s = '' if sample >= 100 else f' TABLESAMPLE SYSTEM ({sample:g})'
    q = f"""SELECT SENSOR_ID, TIMESTAMP_LECTURA, TEMPERATURA_C, HUMEDAD_RELATIVA_PCT,
        PRECIPITACION_MM, CALIDAD_SENAL_DBM
        FROM ARQUI1.CAPA_BRONCE.IOT_TELEMETRIA_CRUDA{s}"""
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
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                 roc_auc_score, average_precision_score, confusion_matrix)
    from sklearn.inspection import permutation_importance
    df, session = cargar()
    df.columns = df.columns.str.lower()
    for col in ['temperatura_c', 'humedad_relativa_pct', 'precipitacion_mm', 'calidad_senal_dbm']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['ts'] = pd.to_datetime(df['timestamp_lectura'], errors='coerce')
    df['riesgo'] = (df.groupby('sensor_id')['temperatura_c'].shift(-LEAD_N) > 30).astype(float)
    print(f'Filas: {len(df)} | tasa riesgo (lead {LEAD_N} lect.): {df["riesgo"].mean():.1%}')
    if LEAD_N >= 1 and float(os.getenv('SAMPLE_PCT', '100')) < 100:
        print('AVISO: con muestra aleatoria el lead no es temporal real; usa SAMPLE_PCT=100')
    df['mes'] = df['ts'].dt.month
    d = df.sort_values(['sensor_id', 'ts']).copy()
    g = d.groupby('sensor_id')
    d['hum_lag_1'] = g['humedad_relativa_pct'].shift(1)
    d['temp_media_6h'] = g['temperatura_c'].shift(1).rolling(6, min_periods=3).mean().reset_index(level=0, drop=True)
    d['temp_std_6h'] = g['temperatura_c'].shift(1).rolling(6, min_periods=3).std().reset_index(level=0, drop=True)
    d['temp_max_6h'] = g['temperatura_c'].shift(1).rolling(6, min_periods=3).max().reset_index(level=0, drop=True)
    d['hum_media_6h'] = g['humedad_relativa_pct'].shift(1).rolling(6, min_periods=3).mean().reset_index(level=0, drop=True)
    d['delta_3h'] = (d['temperatura_c'] - g['temperatura_c'].shift(3)).fillna(0)
    d['humidex'] = d['temperatura_c'] + 0.05 * d['humedad_relativa_pct'] * (d['temperatura_c'] / 30.0)
    d['hora_dia'] = d['ts'].dt.hour
    d['hora_sin'] = np.sin(2 * np.pi * d['hora_dia'] / 24)
    d['hora_cos'] = np.cos(2 * np.pi * d['hora_dia'] / 24)
    d = d.dropna(subset=FEATS + ['riesgo']).copy()
    X_train, X_test, y_train, y_test = train_test_split(
        d[FEATS], d['riesgo'], test_size=0.2, stratify=d['riesgo'], random_state=SEED)
    print(f'Train {len(X_train)} / Test {len(X_test)} | riesgo train {y_train.mean():.1%}')
    n_tr = len(X_train)
    n_trees = int(os.getenv('RF_TREES', '100' if n_tr > 200000 else '300'))
    n_splits = 3 if n_tr > 200000 else 5
    modelos = {
        'Logistica': make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight='balanced', random_state=SEED)),
        'RandomForest': RandomForestClassifier(n_estimators=n_trees, max_depth=14, min_samples_leaf=3, class_weight='balanced_subsample', n_jobs=-1, random_state=SEED),
        'GradBoost': HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=6, class_weight='balanced', random_state=SEED),
    }
    for m in modelos.values():
        m.fit(X_train, y_train)
    print('=== LEADERBOARD (test, umbral 0.5) ===')
    for n, m in modelos.items():
        p, s = m.predict(X_test), m.predict_proba(X_test)[:, 1]
        print(f'{n}: acc={accuracy_score(y_test, p):.3f} prec={precision_score(y_test, p, zero_division=0):.3f} '
              f'rec={recall_score(y_test, p):.3f} F1={f1_score(y_test, p):.3f} AUC={roc_auc_score(y_test, s):.3f}')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    cv = {n: cross_val_score(m, X_train, y_train, cv=skf, scoring='f1') for n, m in modelos.items()}
    campeon = max(cv, key=lambda n: cv[n].mean())
    print(f'Campeon por CV: {campeon} (F1={cv[campeon].mean():.3f})')
    m = modelos[campeon]
    s = m.predict_proba(X_test)[:, 1]
    print('Umbral | Precision | Recall | F1')
    for t in (0.5, 0.4, 0.35, 0.32, 0.3, 0.25):
        p = (s >= t).astype(int)
        print(f'{t:.2f} | {precision_score(y_test, p):.3f} | {recall_score(y_test, p):.3f} | {f1_score(y_test, p):.3f}')
    p_op = (s >= UMBRAL_OP).astype(int)
    print(f'=== REPORTE (umbral operativo {UMBRAL_OP}) ===')
    print(f'prec={precision_score(y_test, p_op):.3f} rec={recall_score(y_test, p_op):.3f} '
          f'F1={f1_score(y_test, p_op):.3f} AUC={roc_auc_score(y_test, s):.3f}')
    print('Matriz [[TN FP][FN TP]]:', confusion_matrix(y_test, p_op).tolist())
    imp = permutation_importance(m, X_test, y_test, n_repeats=5, random_state=SEED, scoring='f1')
    print(pd.Series(imp.importances_mean, index=FEATS).sort_values(ascending=False).round(4).to_string())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, p_op, display_labels=['Normal', 'Riesgo'], cmap='Oranges', colorbar=False, ax=ax[0])
    ax[0].set_title(f'Matriz operativa ({UMBRAL_OP})')
    RocCurveDisplay.from_predictions(y_test, s, ax=ax[1])
    ax[1].plot([0, 1], [0, 1], 'k--', alpha=.4); ax[1].set_title('ROC')
    pd.Series(imp.importances_mean, index=FEATS).sort_values().plot.barh(ax=ax[2]); ax[2].set_title('Importancia')
    plt.tight_layout()
    os.makedirs('docs/img', exist_ok=True)
    plt.savefig('docs/img/riesgo_iot.png', dpi=110)
    print('docs/img/riesgo_iot.png OK')
    if os.getenv('ESCRIBIR', '0') == '1':
        out = X_test.copy()
        out['RIESGO_PRED'] = p_op
        out.columns = [c.upper() for c in out.columns]
        if session is not None:
            session.write_pandas(out, 'PRED_RIESGO_IOT', auto_create_table=True, overwrite=True)
        else:
            out.to_csv('data/pred_riesgo_iot.csv', index=False, encoding='utf-8')
        print('Predicciones guardadas')
if __name__ == '__main__':
    main()
