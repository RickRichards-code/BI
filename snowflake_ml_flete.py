#!/usr/bin/env python3
"""ML regresion en Snowflake: COSTO_FLETE_USD por envio (logistica).
20k envios + 500 transportes. Split CRONOLOGICO 70/15/15 (costos derivan en el
tiempo; aleatorio filtraria futuro). Dummy(media) vs Ridge vs RF vs HGB.
Seleccion en validation (MAE), test final unico. ESCRIBIR=1 guarda PRED_FLETE.
"""
import os
SEED = 42
NUM = ['DISTANCIA_KM', 'DURACION_H', 'CAPACIDAD_TN', 'MES']
CAT = ['TIPO', 'ORIGEN', 'DESTINO', 'PAIS_BANDERA']
def cargar():
    q = """SELECT E.DISTANCIA_KM, DATEDIFF('hour', E.FECHA_SALIDA, E.FECHA_LLEGADA) AS DURACION_H,
        T.CAPACIDAD_TN, EXTRACT(MONTH FROM E.FECHA_SALIDA) AS MES,
        T.TIPO, E.ORIGEN, E.DESTINO, T.PAIS_BANDERA, E.COSTO_FLETE_USD, E.FECHA_SALIDA
        FROM ARQUI1.CAPA_BRONCE.ENVIO E
        LEFT JOIN ARQUI1.CAPA_BRONCE.TRANSPORTE T ON E.ID_TRANSPORTE = T.ID_TRANSPORTE
        WHERE E.COSTO_FLETE_USD > 0 AND E.DISTANCIA_KM > 0"""
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
    print('Origen: connector local')
    return pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description]), None
def main():
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    df, session = cargar()
    df.columns = df.columns.str.lower()
    num = [x.lower() for x in NUM]
    cat = [x.lower() for x in CAT]
    for col in num + ['costo_flete_usd']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in cat:
        df[col] = df[col].fillna('FALTA').astype(str)
    df['fecha_salida'] = pd.to_datetime(df['fecha_salida'], errors='coerce')
    df = df.dropna(subset=num + ['costo_flete_usd']).sort_values('fecha_salida').reset_index(drop=True)
    print(f'Filas: {len(df)} | {df["fecha_salida"].min()} -> {df["fecha_salida"].max()}')
    print(df[['costo_flete_usd', 'distancia_km', 'duracion_h']].describe().round(1).to_string())
    n = len(df)
    tr, va, te = df.iloc[:int(n * 0.7)], df.iloc[int(n * 0.7):int(n * 0.85)], df.iloc[int(n * 0.85):]
    Xtr, ytr, Xva, yva, Xte, yte = tr[num + cat], tr['costo_flete_usd'], va[num + cat], va['costo_flete_usd'], te[num + cat], te['costo_flete_usd']
    print(f'train={len(tr)} val={len(va)} test={len(te)} (cronologico)')
    pre = ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), num),
        ('cat', make_pipeline(SimpleImputer(strategy='constant', fill_value='FALTA'),
                               OneHotEncoder(handle_unknown='ignore')), cat)])
    cand = {
        'Dummy': DummyRegressor(strategy='mean'),
        'Ridge': make_pipeline(pre, Ridge()),
        'RF': RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        'HGB': HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=SEED),
    }
    def met(y, p):
        return {'MAE': mean_absolute_error(y, p), 'RMSE': float(np.sqrt(mean_squared_error(y, p))),
                'R2': r2_score(y, p), 'MAPE': float(np.mean(np.abs(y - p) / np.maximum(y, 1)) * 100)}
    print('=== VALIDATION (MAE) ===')
    best, bn = 1e18, ''
    for name, m in cand.items():
        if name in ('RF', 'HGB'):
            Xe = pd.get_dummies(Xtr[num + cat], dummy_na=False)
            Xv = pd.get_dummies(Xva[num + cat], dummy_na=False).reindex(columns=Xe.columns, fill_value=0)
            med = Xe.median()
            Xe, Xv = Xe.fillna(med), Xv.fillna(med)
            m.fit(Xe, ytr)
            r = met(yva.values, m.predict(Xv))
        else:
            m.fit(Xtr, ytr)
            r = met(yva.values, m.predict(Xva))
        print(f'{name}: MAE={r["MAE"]:.2f} RMSE={r["RMSE"]:.2f} R2={r["R2"]:.3f} MAPE={r["MAPE"]:.1f}%')
        if r['MAE'] < best:
            best, bn, bmet = r['MAE'], name, r
    print(f'Campeon: {bn}')
    print('=== REPORTE (TEST unico, futuro real) ===')
    m = cand[bn]
    if bn in ('RF', 'HGB'):
        Xe = pd.get_dummies(Xtr[num + cat], dummy_na=False)
        Xt = pd.get_dummies(Xte[num + cat], dummy_na=False).reindex(columns=Xe.columns, fill_value=0)
        Xv2 = pd.get_dummies(Xva[num + cat], dummy_na=False).reindex(columns=Xe.columns, fill_value=0)
        med = Xe.median()
        Xe, Xt, Xv2 = Xe.fillna(med), Xt.fillna(med), Xv2.fillna(med)
        m.fit(pd.concat([Xe, Xv2]),
              pd.concat([ytr, yva]))
        p = m.predict(Xt)
    else:
        m.fit(pd.concat([Xtr, Xva]), pd.concat([ytr, yva]))
        p = m.predict(Xte)
    r = met(yte.values, p)
    print(f'Modelo={bn} MAE={r["MAE"]:.2f} USD RMSE={r["RMSE"]:.2f} R2={r["R2"]:.3f} MAPE={r["MAPE"]:.1f}%')
    print(f'Mejora vs Dummy: {(1 - r["MAE"] / met(yte.values, np.full(len(yte), ytr.mean()))["MAE"]) * 100:.1f}% menos error')
    if os.getenv('ESCRIBIR', '0') == '1':
        out = te[['fecha_salida']].copy()
        out['pred_usd'] = np.round(p, 2)
        out.columns = [c.upper() for c in out.columns]
        if session is not None:
            session.write_pandas(out, 'PRED_FLETE', auto_create_table=True, overwrite=True)
        else:
            out.to_csv('data/pred_flete.csv', index=False, encoding='utf-8')
        print('Predicciones guardadas')
if __name__ == '__main__':
    main()
