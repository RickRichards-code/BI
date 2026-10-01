#!/usr/bin/env python3
"""Asignacion optima de recepciones a silos (Oro + Bronce).
Predice merma esperada por lote y resuelve asignacion lote->silo con
programacion lineal entera (PuLP/CBC): minimiza merma + desbalance,
sujeto a capacidad y stock actual.
Uso:  python mineria_asignacion_silos.py            (ultimos 7 dias, 85 silos)
      DIAS=3 LOTES_MAX=200 python mineria_asignacion_silos.py  (prueba)
Requiere SNOWFLAKE_PASSWORD. Guarda data/asignacion_optima.csv
"""
import os
import numpy as np
import pandas as pd
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
def leer(query):
    c = conectar(); cur = c.cursor(); cur.execute(query)
    df = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
    df.columns = df.columns.str.lower()
    return df
def main():
    dias = int(os.getenv('DIAS', '7'))
    nmax = int(os.getenv('LOTES_MAX', '1000000'))
    lotes = leer(f"""SELECT ID_RECEPCION, ID_SILO AS silo_real, PESO_BRUTO_TN - TARA_TN AS PESO_NETO,
                            HUMEDAD_PCT, IMPUREZAS_PCT
                     FROM AIRBYTE_DATABASE.BRONCE.RECEPCION_GRANO
                     WHERE FECHA_HORA >= (SELECT DATEADD(day, -{dias}, MAX(FECHA_HORA)) FROM AIRBYTE_DATABASE.BRONCE.RECEPCION_GRANO
                                          WHERE FECHA_HORA BETWEEN '2020-01-01' AND '2027-01-01')
                       AND FECHA_HORA BETWEEN '2020-01-01' AND '2027-01-01'
                       AND PESO_BRUTO_TN - TARA_TN BETWEEN 1 AND 200
                     LIMIT {nmax}""")
    for col in ['peso_neto', 'humedad_pct', 'impurezas_pct']:
        lotes[col] = pd.to_numeric(lotes[col], errors='coerce')
    lotes = lotes.dropna().copy()
    if len(lotes) == 0:
        raise SystemExit('Sin lotes en la ventana: ajusta DIAS')
    print(f'Lotes a asignar: {len(lotes)} (ultimos {dias} dias de dato cuerdo)')
    silos = leer("""SELECT s.ID_SILO, s.CAPACIDAD_TN, s.TEMPERATURA_CONTROL,
                           COALESCE(i.STOCK_TN, 0) AS STOCK_TN
                    FROM AIRBYTE_DATABASE.BRONCE.SILO s
                    LEFT JOIN (SELECT ID_SILO, SUM(STOCK_TN) AS STOCK_TN
                               FROM AIRBYTE_DATABASE.BRONCE.INVENTARIO_GRANO GROUP BY 1) i
                      ON i.ID_SILO = s.ID_SILO""")
    silos['capacidad_tn'] = pd.to_numeric(silos['capacidad_tn'], errors='coerce')
    silos['stock_tn'] = pd.to_numeric(silos['stock_tn'], errors='coerce')
    # Inventario inconsistente con capacidad (sintetico: stocks 10-100x). Se planifica
    # sobre capacidad nominal; el stock solo pondera el factor de llenado (capado a 1).
    silos['libre_tn'] = silos['capacidad_tn'].clip(lower=0)
    silos['fill'] = (silos['stock_tn'] / silos['capacidad_tn'].replace(0, np.nan)).clip(0, 1).fillna(0)
    silos = silos[silos['libre_tn'] > 0].copy()
    print(f'Silos con espacio: {len(silos)}')
    # Merma esperada lote x silo: base humedad/impurezas x factor silo (sin control +20%, lleno>90% +10%)
    hum = lotes['humedad_pct'].values / 100.0
    imp = lotes['impurezas_pct'].values / 100.0
    peso = lotes['peso_neto'].values
    base = peso * (np.clip(hum - 0.14, 0, None) + np.clip(imp - 0.01, 0, None))
    f_silo = (1 + 0.2 * (~silos['temperatura_control'].fillna(False).astype(bool)).values
              + 0.1 * (silos['fill'].values > 0.9).astype(float))
    costo = base[:, None] * f_silo[None, :]
    try:
        import pulp
    except ImportError:
        raise SystemExit('Falta PuLP: pip install pulp')
    L, S = len(lotes), len(silos)
    prob = pulp.LpProblem('silos', pulp.LpMinimize)
    x = {(l, s): pulp.LpVariable(f'x_{l}_{s}', cat='Binary') for l in range(L) for s in range(S)}
    prob += pulp.lpSum(costo[l, s] * x[l, s] for l in range(L) for s in range(S))
    for l in range(L):
        prob += pulp.lpSum(x[l, s] for s in range(S)) == 1
    cap = silos['libre_tn'].values
    for s in range(S):
        prob += pulp.lpSum(lotes['peso_neto'].values[l] * x[l, s] for l in range(L)) <= cap[s]
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    print('Estado:', pulp.LpStatus[prob.status])
    asig = np.array([[int(pulp.value(x[l, s])) for s in range(S)] for l in range(L)])
    lotes['silo_optimo'] = (asig * silos['id_silo'].values[None, :]).sum(axis=1)
    opt = float((costo * asig).sum())
    # Baseline con EL MISMO factor silo del silo realmente usado (comparacion justa)
    fmap = dict(zip(silos['id_silo'].values, f_silo))
    f_real = lotes['silo_real'].map(fmap).fillna(1.0).values
    merma_real = float((base * f_real).sum())
    print(f'Merma esperada baseline (asignacion real): {merma_real:,.1f} TN')
    print(f'Merma esperada optima:                    {opt:,.1f} TN')
    print(f'Ahorro estimado: {(1 - opt / max(merma_real, 1)) * 100:.1f}% ({merma_real - opt:,.1f} TN)')
    fill = pd.DataFrame({'silo': silos['id_silo'].values,
                         'carga_tn': [(lotes.loc[lotes['silo_optimo'] == s, 'peso_neto'].sum()) for s in silos['id_silo'].values]})
    print('Top silos cargados:'); print(fill.sort_values('carga_tn', ascending=False).head(5).to_string(index=False))
    os.makedirs('data', exist_ok=True)
    lotes[['id_recepcion', 'silo_optimo']].to_csv('data/asignacion_optima.csv', index=False, encoding='utf-8')
    print('Guardado data/asignacion_optima.csv')
if __name__ == '__main__':
    main()
