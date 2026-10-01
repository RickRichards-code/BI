#!/usr/bin/env python3
"""Fraude y fuga en bascula (Bronce): score 0-100 + tipologias + agregados.
NO binario: cada recepcion recibe score continuo, 1 tipologia principal y severidad.
Uso:  python mineria_fraude_bascula.py            (SAMPLE_PCT=100 por defecto)
      SAMPLE_PCT=1 python mineria_fraude_bascula.py  (prueba rapida 1%)
Requiere SNOWFLAKE_PASSWORD en entorno. Guarda top-5000 y agregados en data/.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
SEED = 42
BENFORD = np.array([np.log10(1 + 1 / d) for d in range(1, 10)])
def q_main(sample):
    s = '' if float(sample) >= 100 else f' TABLESAMPLE SYSTEM ({float(sample)})'
    return f"""
SELECT r.ID_RECEPCION, r.FECHA_HORA, r.ID_SILO, r.ID_CONTRATO,
       r.PESO_BRUTO_TN, r.TARA_TN, r.HUMEDAD_PCT, r.IMPUREZAS_PCT,
       r.CALIDAD_GRANO, r.NUMERO_REMISION,
       c.ID_PRODUCTOR, c.PRECIO_ACORDADO_USD_TN, c.VOLUMEN_COMPROMETIDO_TN, c.PRECIO_REFERENCIA_CBOT,
       COALESCE(lab.N_LAB, 0) AS N_LAB
FROM AIRBYTE_DATABASE.BRONCE.RECEPCION_GRANO r{s}
LEFT JOIN AIRBYTE_DATABASE.BRONCE.CONTRATO_COMPRA c ON r.ID_CONTRATO = c.ID_CONTRATO
LEFT JOIN (SELECT a.ID_RECEPCION, COUNT(*) AS N_LAB
           FROM AIRBYTE_DATABASE.BRONCE.ASIGNACION_GRANO a
           JOIN AIRBYTE_DATABASE.BRONCE.CONTROL_CALIDAD q ON q.ID_LOTE = a.ID_LOTE
           GROUP BY 1) lab ON lab.ID_RECEPCION = r.ID_RECEPCION"""
def cargar(sample):
    import snowflake.connector
    pwd = os.getenv('SNOWFLAKE_PASSWORD', '').strip().strip('\'"')
    if not pwd:
        raise SystemExit('Sin SNOWFLAKE_PASSWORD: export SNOWFLAKE_PASSWORD=\'clave\' (este script no usa CSV: son 1.5M filas)')
    kw = dict(user=os.getenv('SNOWFLAKE_USER', 'ENRRIQUE'), password=pwd,
              account=os.getenv('SNOWFLAKE_ACCOUNT', 'AVBVHGL-WZ57062'),
              database='AIRBYTE_DATABASE', warehouse='COMPUTE_WH', login_timeout=60)
    if os.getenv('SNOWFLAKE_ROLE'):
        kw['role'] = os.getenv('SNOWFLAKE_ROLE')
    c = snowflake.connector.connect(**kw)
    cur = c.cursor(); cur.execute(q_main(sample))
    df = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
    df.columns = df.columns.str.lower()
    for col in ['peso_bruto_tn', 'tara_tn', 'humedad_pct', 'impurezas_pct', 'precio_acordado_usd_tn',
                'volumen_comprometido_tn']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f'Filas: {len(df)} (muestra {sample}%)')
    return df
def benford_p(df):
    """p-valor chi-cuadrado Benford por estrato silo-mes; devuelve -log10(p) por fila."""
    d1 = df['peso_neto'].abs().astype(str).str.extract(r'([1-9])')[0].astype(float)
    df['_d1'] = d1
    df['_ym'] = df['fecha_hora'].dt.strftime('%Y-%m')
    sig = pd.Series(0.0, index=df.index)
    for (_, _), g in df.groupby(['id_silo', '_ym']):
        obs = g['_d1'].value_counts(normalize=True).reindex(range(1, 10), fill_value=0).values
        esp = BENFORD * len(g)
        chi = (((obs * len(g) - esp) ** 2) / esp).sum()
        from scipy.stats import chi2 as _chi2
        p = _chi2.sf(chi, 8)
        sig.loc[g.index] = min(-np.log10(max(p, 1e-12)), 6.0)
    return sig
def main():
    sample = float(os.getenv('SAMPLE_PCT', '100'))
    full = sample >= 100
    df = cargar(sample)
    df['fecha_hora'] = pd.to_datetime(df['fecha_hora'], errors='coerce')
    df['peso_neto'] = df['peso_bruto_tn'] - df['tara_tn']
    df['tara_ratio'] = df['tara_tn'] / df['peso_bruto_tn'].replace(0, np.nan)
    df['hora_dec'] = df['fecha_hora'].dt.hour + df['fecha_hora'].dt.minute / 60
    df['es_madrugada'] = ((df['hora_dec'] < 5) | (df['hora_dec'] > 23)).astype(float)
    df['es_domingo'] = (df['fecha_hora'].dt.dayofweek == 6).astype(float)
    df['dup_remision'] = df.duplicated('numero_remision', keep=False).astype(float)
    yr = df['fecha_hora'].dt.year
    df['s_dup'] = df['dup_remision'] * 0.6
    df['s_date'] = ((yr.lt(2000) | yr.gt(2027))).astype(float) * 0.6
    print(f"diag: dup_remision={df['dup_remision'].mean():.1%} contrato_null={df['id_contrato'].isna().mean():.1%} fecha_rara={df['s_date'].mean():.1%}")
    # Reglas duras -> 100 directo (solo combinaciones graves o fisica imposible)
    hard_doc = ((df['dup_remision'] == 1) & (df['s_date'] > 0) & df['id_contrato'].isna()).astype(float)
    hard_fis = ((df['peso_neto'] <= 0) | (df['tara_tn'] > df['peso_bruto_tn']) | (df['peso_neto'] > 200)).astype(float)
    # z robusto por silo-mes (fisica)
    g = df.groupby(['id_silo', df['fecha_hora'].dt.strftime('%Y-%m')])
    for col in ['humedad_pct', 'impurezas_pct', 'peso_neto']:
        med = g[col].transform('median')
        mad = (g[col].transform(lambda s: (s - s.median()).abs().median()) + 1e-9)
        df[f'z_{col}'] = ((df[col] - med) / (1.4826 * mad)).abs().fillna(0)
    df['s_fisica'] = df[['z_humedad_pct', 'z_impurezas_pct']].max(axis=1).clip(0, 6) / 6
    # Rendimiento TN/ha por productor (necesita superficie: query pequena)
    try:
        import snowflake.connector
        c2 = snowflake.connector.connect(user=os.getenv('SNOWFLAKE_USER', 'ENRRIQUE'),
            password=os.getenv('SNOWFLAKE_PASSWORD', '').strip().strip('\'"'),
            account=os.getenv('SNOWFLAKE_ACCOUNT', 'AVBVHGL-WZ57062'), database='AIRBYTE_DATABASE', warehouse='COMPUTE_WH')
        cur = c2.cursor()
        cur.execute('SELECT ID_PRODUCTOR, SUM(SUPERFICIE_HA) FROM AIRBYTE_DATABASE.BRONCE.PREDIO_LOTE GROUP BY 1')
        sup = dict(cur.fetchall())
        vol = df.groupby('id_productor')['peso_neto'].sum()
        yld = vol / pd.Series(sup)
        df['s_rend'] = (yld.reindex(df['id_productor']).values / 8.0).clip(0, 1)  # techo 8 TN/ha
        df['s_rend'] = df['s_rend'].fillna(0)
    except Exception as e:
        print('Sin superficie:', e); df['s_rend'] = 0.0
    # Contrato: sobre-entrega acumulada (solo muestra completa)
    if full:
        df = df.sort_values('fecha_hora')
        cum = df.groupby('id_contrato')['peso_neto'].cumsum()
        comp = df['volumen_comprometido_tn'].replace(0, np.nan)
        df['s_contrato'] = ((cum / comp) - 1).clip(lower=0).fillna(0).clip(0, 1)
        df['s_contrato'] += (df['id_contrato'].isna() & (df['peso_neto'] > 50)).astype(float) * 0.3
        df['s_contrato'] += (df['id_contrato'].isna() & (df['peso_neto'] <= 50)).astype(float) * 0.1
    else:
        df['s_contrato'] = df['id_contrato'].isna().astype(float) * 0.2
        print('(muestra: sobre-entrega desactivada)')
    # Laboratorio ausente con calidad premium declarada
    df['s_lab'] = ((df['n_lab'] == 0)).astype(float) * 0.4
    # Horario raro
    df['s_hora'] = (df['es_madrugada'] * 0.7 + df['es_domingo'] * 0.3).clip(0, 1)
    # Benford por silo-mes
    try:
        df['s_benford'] = benford_p(df) / 6.0
    except ImportError:
        print('Sin scipy: Benford aproximado por digito raro')
        d1 = df['peso_neto'].abs().astype(str).str.extract(r'([1-9])')[0].astype(float)
        fr = d1.value_counts(normalize=True)
        df['s_benford'] = d1.map(lambda x: abs(fr.get(x, 0) - BENFORD[int(x) - 1])).fillna(0).clip(0, 0.5) * 2
    # IsolationForest
    feats = ['peso_neto', 'tara_ratio', 'hora_dec', 'humedad_pct', 'impurezas_pct',
             'z_peso_neto', 's_rend', 's_contrato', 'es_madrugada', 'es_domingo']
    Xm = df[feats].fillna(df[feats].median()).values
    Xs = StandardScaler().fit_transform(Xm)
    iso = IsolationForest(n_estimators=100, max_samples=256, contamination=0.02, random_state=SEED, n_jobs=-1).fit(Xs)
    df['s_iso'] = ((0 - iso.score_samples(Xs)) / 0.5).clip(0, 1)
    # Score 0-100 (DOCUMENTAL aporta como familia puntuada, no solo hard)
    df['s_doc'] = df[['s_contrato', 's_dup', 's_date']].max(axis=1)
    df['score'] = (15 * df['s_benford'] + 25 * df['s_iso'] + 15 * df['s_rend'] + 10 * df['s_contrato']
                   + 5 * df['s_lab'] + 10 * df['s_hora'] + 15 * df['s_fisica'] + 5 * df['z_peso_neto'].clip(0, 6) / 6
                   + 10 * df['s_doc']) / 1.1
    df.loc[hard_doc == 1, 'score'] = 100.0
    df.loc[hard_fis == 1, 'score'] = 100.0
    df['score'] = df['score'].clip(0, 100).round(1)
    fam = {'FISICA': df['s_fisica'], 'DOCUMENTAL': df['s_doc'],
           'RENDIMIENTO': df['s_rend'], 'ESTADISTICA': df[['s_benford', 's_iso']].max(axis=1), 'HORARIO': df['s_hora']}
    df['tipologia'] = pd.DataFrame(fam).idxmax(axis=1)
    df.loc[hard_doc == 1, 'tipologia'] = 'DOCUMENTAL'
    df.loc[hard_fis == 1, 'tipologia'] = 'FISICA'
    q99, q90, q50 = df['score'].quantile([0.99, 0.90, 0.50])
    print(f'cortes percentil: critico>={q99:.1f} alto>={q90:.1f} medio>={q50:.1f}')
    df['severidad'] = pd.cut(df['score'], [-0.1, q50, q90, q99, 100.1], labels=['bajo', 'medio', 'alto', 'critico'])
    print('=== TIPOLOGIAS ==='); print(df['tipologia'].value_counts().to_string())
    print('=== SEVERIDAD ==='); print(df['severidad'].value_counts().to_string())
    top = df.nlargest(20, 'score')[['id_recepcion', 'fecha_hora', 'id_silo', 'id_contrato', 'peso_neto', 'score', 'tipologia', 'severidad']]
    print('=== TOP-20 ==='); print(top.to_string())
    for col, name in [('id_silo', 'silo'), ('id_contrato', 'contrato')]:
        ag = df.groupby(col)['score'].agg(['mean', 'max', 'count']).round(1).sort_values('mean', ascending=False).head(10)
        ag.to_csv(f'data/riesgo_{name}.csv', encoding='utf-8'); print(f'--- top {name} ---'); print(ag.to_string())
    dfm = df.copy(); dfm['mes'] = dfm['fecha_hora'].dt.strftime('%Y-%m')
    agm = dfm.groupby('mes')['score'].agg(['mean', 'max', 'count']).round(1)
    agm.to_csv('data/riesgo_mes.csv', encoding='utf-8'); print(agm.tail(6).to_string())
    df.nlargest(5000, 'score').to_csv('data/alerta_top.csv', index=False, encoding='utf-8')
    print('Guardados data/alerta_top.csv + riesgo_silo/contrato/mes.csv')
if __name__ == '__main__':
    main()
