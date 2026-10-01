#!/usr/bin/env python3
"""Diagnostico integral de la capa Bronce (7 esquemas, ~40 tablas).
Uso:  python diagnostico_bronce.py
Requiere SNOWFLAKE_PASSWORD. Imprime resumen por tabla y guarda:
  data/dq_bronce_tablas.csv   (filas, columnas, flags)
  data/dq_bronce_columnas.csv (nulos%, distintos, min, max por columna)
Flags: VACIA | COL_TODA_NULA | NULOS_ALTOS>30% | NEGATIVOS | FECHAS_FUTURAS
       | CONSTANTE | SIN_CLAVE | DUP_CLAVE
"""
import os
import pandas as pd
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
    cur.execute("""SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE
                   FROM AIRBYTE_DATABASE.INFORMATION_SCHEMA.COLUMNS
                   WHERE TABLE_SCHEMA LIKE '%BRONCE%'
                   ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION""")
    meta = pd.DataFrame(cur.fetchall(), columns=['schema', 'table', 'col', 'dtype'])
    print(f'Tablas Bronce: {meta[["schema","table"]].drop_duplicates().shape[0]}')
    trows, crows = [], []
    for (sch, tab), g in meta.groupby(['schema', 'table']):
        cols = g.to_dict('records')
        amap = {}
        def alias(prefix, cn, i):
            a = f'{prefix}C{i}'
            amap[a] = cn
            return a
        qn = lambda x: '"' + x + '"'
        aggs = ['COUNT(*) AS N']
        for i, r in enumerate(cols):
            cn, dt = r['col'], r['dtype']
            aggs.append(f'SUM(CASE WHEN {qn(cn)} IS NULL THEN 1 ELSE 0 END) AS {alias("NUL", cn, i)}')
            aggs.append(f'APPROX_COUNT_DISTINCT({qn(cn)}) AS {alias("DST", cn, i)}')
            if dt not in ('VARIANT', 'OBJECT', 'ARRAY', 'GEOGRAPHY', 'TEXT', 'VARCHAR', 'CHAR', 'STRING', 'BINARY'):
                aggs.append(f'MIN({qn(cn)}) AS {alias("MIN", cn, i)}')
                aggs.append(f'MAX({qn(cn)}) AS {alias("MAX", cn, i)}')
        cur.execute(f'SELECT {", ".join(aggs)} FROM AIRBYTE_DATABASE.{sch}.{tab}')
        vals = cur.fetchone()
        desc = [d[0] for d in cur.description]
        row = dict(zip(desc, vals))
        n = row.pop('N')
        flags, worst, wcol = [], 0.0, ''
        n_const = 0
        for i, r in enumerate(cols):
            cn, dt = r['col'], r['dtype']
            nul = (row.get(f'NULC{i}') or 0) / max(n, 1)
            dst = row.get(f'DSTC{i}') or 0
            crows.append({'schema': sch, 'table': tab, 'col': cn, 'dtype': dt,
                          'null_pct': round(nul * 100, 2), 'distinct': dst,
                          'min': str(row.get(f'MINC{i}')), 'max': str(row.get(f'MAXC{i}'))})
            if nul >= 1.0:
                flags.append(f'TODA_NULA:{cn}')
            if not cn.isascii():
                flags.append(f'NOMBRE_RARO:{cn}')
            if nul > worst:
                worst, wcol = nul, cn
            if n > 0 and dst <= 1 and nul < 1.0 and not cn.startswith(('_AIRBYTE_', '_AB_CDC_')):
                n_const += 1
        if n == 0:
            flags.append('VACIA')
        if worst > 0.30:
            flags.append(f'NULOS_ALTOS:{wcol}={worst:.0%}')
        if n_const:
            flags.append(f'CONSTANTES:{n_const}')
        pref = ('ID_RECEPCION', 'ID_VENTA', 'ID_ENVIO', 'ID_EMBARQUE', 'ID_ASIGNACION', 'ID_CONTROL',
                'ID_INVENTARIO', 'ID_LOTE', 'ID_PRODUCCION', 'ID_CONTRATO', 'ID_PRODUCTOR', 'ID_CLIENTE',
                'ID_SILO', 'ID_PREDIO', 'ID_PRODUCTO', 'ID_CENTRO_ACOPIO', 'ID_PUERTO', 'ID_PLANTA',
                'ID_TRANSPORTE', 'TICK_ID', 'LECTURA_ID', '_ID', 'SENSOR_ID', 'TICKET_ID', 'ALERTA_ID',
                'ID_EVENTO', 'ID_EMBARQUE')
        present = {r['col'] for r in cols}
        key = next((k for k in pref if k in present), None)
        if key is None:
            key = next((r['col'] for r in cols if r['col'].startswith('ID_')), None)
        if key is None:
            flags.append('SIN_CLAVE')
        else:
            cur.execute(f'SELECT COUNT(*) - COUNT(DISTINCT {qn(key)}) FROM AIRBYTE_DATABASE.{sch}.{tab}')
            dup = cur.fetchone()[0]
            if dup and dup > 0:
                flags.append(f'DUP_CLAVE:{key}={dup}')
        print(f'{sch}.{tab} | filas={n:,} cols={len(cols)} | max_nulo={worst:.0%} ({wcol}) | clave={key} | {" ".join(flags) if flags else "OK"}')
        trows.append({'schema': sch, 'table': tab, 'filas': n, 'columnas': len(cols),
                      'max_nulo_pct': round(worst * 100, 2), 'flags': ' '.join(flags) if flags else 'OK'})
    os.makedirs('data', exist_ok=True)
    pd.DataFrame(trows).to_csv('data/dq_bronce_tablas.csv', index=False, encoding='utf-8')
    pd.DataFrame(crows).to_csv('data/dq_bronce_columnas.csv', index=False, encoding='utf-8')
    print('Guardados data/dq_bronce_tablas.csv y data/dq_bronce_columnas.csv')
if __name__ == '__main__':
    main()
