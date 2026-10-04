#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ML viable segun DGP: A) precio unitario venta (regresion) B) merma pre-recepcion (clasificacion sin lab).
Snowpark-first (sin credenciales en Snowflake). SAMPLE_PCT rige B (A usa 54k completas).
"""
import os
SEED = 42
def cargar(q):
    try:
        from snowflake.snowpark.context import get_active_session
    except ImportError:
        return _cargar_local(q)
    session = get_active_session()
    print('Origen: sesion Snowflake (Snowpark), sin credenciales')
    return session.sql(q).to_pandas(), session
def _cargar_local(q):
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
    return pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description]), None
def main():
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
    print('########## A. PRECIO UNITARIO VENTA ##########')
    qa = """SELECT V.CANTIDAD_TN, V.PRIMA_BASIS_USD, V.PRECIO_CBOT_REFERENCIA, V.PRECIO_UNITARIO_USD,
        V.ID_PRODUCTO, C.SEGMENTO, C.PAIS, CV.INCOTERM, CV.PUERTO_EMBARQUE,
        EXTRACT(MONTH FROM V.FECHA) AS MES, EXTRACT(YEAR FROM V.FECHA) AS ANIO
        FROM ARQUI1.CAPA_BRONCE.VENTA V
        LEFT JOIN ARQUI1.CAPA_BRONCE.CONTRATO_VENTA CV ON V.ID_CONTRATO_VENTA = CV.ID_CONTRATO_VENTA
        LEFT JOIN ARQUI1.CAPA_BRONCE.CLIENTE C ON CV.ID_CLIENTE = C.ID_CLIENTE
        WHERE V.FECHA BETWEEN '2020-01-01' AND '2027-01-01'
          AND V.CANTIDAD_TN < 5000 AND V.PRECIO_UNITARIO_USD < 2000"""
    df, _ = (None, None)
    # carga directa (misma funcion, query propia)
    try:
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        df = session.sql(qa).to_pandas()
        print('Origen: sesion Snowflake (Snowpark), sin credenciales')
    except Exception:
        df, _ = _cargar_local(qa)
    df.columns = df.columns.str.lower()
    num = ['cantidad_tn', 'prima_basis_usd', 'precio_cbot_referencia', 'mes', 'anio']
    cat = ['id_producto', 'segmento', 'pais', 'incoterm', 'puerto_embarque']
    for col in num + ['precio_unitario_usd']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in cat:
        df[col] = df[col].fillna('FALTA').astype(str)
    df = df.dropna(subset=num + ['precio_unitario_usd']).copy()
    print(f'Filas: {len(df)}')
    d = df.sort_values(['anio', 'mes']).reset_index(drop=True)
    n = len(d)
    tr, va, te = d.iloc[:int(n * 0.7)], d.iloc[int(n * 0.7):int(n * 0.85)], d.iloc[int(n * 0.85):]
    pre = ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), num),
        ('cat', make_pipeline(SimpleImputer(strategy='constant', fill_value='FALTA'),
                               OneHotEncoder(handle_unknown='ignore')), cat)])
    ts = TimeSeriesSplit(n_splits=3)
    import numpy as _np
    from sklearn.linear_model import RidgeCV as _R
    cand = {'Ridge': make_pipeline(pre, _R(alphas=_np.logspace(-1, 5, 20), cv=ts)),
            'RF': make_pipeline(pre, RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=SEED))}
    print('--- VALIDATION (MAE) ---')
    best, bn = 1e18, ''
    for name, m in cand.items():
        m.fit(tr[num + cat], tr['precio_unitario_usd'])
        p = m.predict(va[num + cat])
        mae = mean_absolute_error(va['precio_unitario_usd'], p)
        print(f'{name}: MAE={mae:.2f} R2={r2_score(va["precio_unitario_usd"], p):.3f}')
        if mae < best:
            best, bn = mae, name
    m = cand[bn]
    m.fit(pd.concat([tr[num + cat], va[num + cat]]), pd.concat([tr['precio_unitario_usd'], va['precio_unitario_usd']]))
    p = m.predict(te[num + cat])
    print(f'=== REPORTE A: {bn} MAE={mean_absolute_error(te["precio_unitario_usd"], p):.2f} '
          f'R2={r2_score(te["precio_unitario_usd"], p):.3f} MAPE={np.mean(np.abs(te["precio_unitario_usd"] - p) / te["precio_unitario_usd"]) * 100:.1f}% ===')
    print('########## B. MERMA PRE-RECEPCION (sin laboratorio) ##########')
    sample = float(os.getenv('SAMPLE_PCT', '20'))
    s = '' if sample >= 100 else f' TABLESAMPLE SYSTEM ({sample:g})'
    qb = f"""SELECT R.ID_CONTRATO, C.ID_PRODUCTOR, P.TIPO_PRODUCTOR, P.AREA_TOTAL_HA,
        EXTRACT(MONTH FROM R.FECHA_HORA) AS MES, R.ID_SILO,
        (R.PESO_BRUTO_TN - R.TARA_TN) * ((GREATEST(R.HUMEDAD_PCT - 14, 0)) / 100 + (GREATEST(R.IMPUREZAS_PCT - 1, 0)) / 100) AS MERMA
        FROM ARQUI1.CAPA_BRONCE.RECEPCION_GRANO R{s}
        LEFT JOIN ARQUI1.CAPA_BRONCE.CONTRATO_COMPRA C ON R.ID_CONTRATO = C.ID_CONTRATO
        LEFT JOIN ARQUI1.CAPA_BRONCE.PRODUCTOR P ON C.ID_PRODUCTOR = P.ID_PRODUCTOR
        WHERE R.PESO_BRUTO_TN - R.TARA_TN BETWEEN 1 AND 200 AND R.FECHA_HORA BETWEEN '2020-01-01' AND '2027-01-01'"""
    try:
        from snowflake.snowpark.context import get_active_session
        g = session.sql(qb).to_pandas()
    except Exception:
        g, _ = _cargar_local(qb)
    g.columns = g.columns.str.lower()
    for col in ['area_total_ha', 'mes', 'id_silo', 'merma']:
        g[col] = pd.to_numeric(g[col], errors='coerce')
    g['tipo_productor'] = g['tipo_productor'].fillna('FALTA').astype(str)
    umb = g['merma'].quantile(0.85)
    g['y'] = (g['merma'] > umb).astype(int)
    print(f'Filas: {len(g)} | umbral p85={umb:.2f} TN | tasa positiva: {g["y"].mean():.1%}')
    num2 = ['area_total_ha', 'mes']
    cat2 = ['tipo_productor', 'id_silo']
    g = g.dropna(subset=num2 + ['y']).copy()
    pre2 = ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), num2),
        ('cat', make_pipeline(SimpleImputer(strategy='constant', fill_value='FALTA'),
                               OneHotEncoder(handle_unknown='ignore', sparse_output=False)), cat2)])
    from sklearn.model_selection import train_test_split as tts
    Xtr, Xte, ytr, yte = tts(g[num2 + cat2], g['y'], test_size=0.3, stratify=g['y'], random_state=SEED)
    for name, m in {'LogReg': make_pipeline(pre2, LogisticRegression(max_iter=500, class_weight='balanced', random_state=SEED)),
                    'RF': make_pipeline(pre2, RandomForestClassifier(n_estimators=200, min_samples_leaf=10, class_weight='balanced_subsample', n_jobs=-1, random_state=SEED)),
                    'HGB': make_pipeline(pre2, HistGradientBoostingClassifier(max_iter=200, random_state=SEED))}.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        s_ = m.predict_proba(Xte)[:, 1]
        print(f'{name}: acc={accuracy_score(yte, p):.3f} F1={f1_score(yte, p, zero_division=0):.3f} AUC={roc_auc_score(yte, s_):.3f}')
    print(f'=== REPORTE B: umbral={umb:.2f} (ver F1/AUC arriba; baseline=mayoria={1 - g["y"].mean():.3f} acc) ===')
if __name__ == '__main__':
    main()
