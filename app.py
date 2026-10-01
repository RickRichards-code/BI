"""App de despliegue CRISP-DM: explora los 8 frentes desde CSVs versionados (sin Snowflake).
Uso local:  streamlit run app.py
En Lightning: se abre como app web del Studio.
"""
import os
import pandas as pd
import streamlit as st
st.set_page_config(page_title='Gravetal · Mineria', layout='wide')
st.title('Gravetal · Mineria operativa (8 frentes)')
frente = st.sidebar.radio('Frente', ['Segmentos', 'Fraude', 'Regimenes', 'Asignacion', 'Mercado', 'Jerarquico'])

def csv(path):
    if not os.path.exists(path):
        st.warning(f'Falta {path}: corre el script y commitea el CSV.')
        st.stop()
    return pd.read_csv(path)

if frente == 'Segmentos':
    st.header('1. Segmentos de productores')
    seg = csv('data/productores_segmentos_v2.csv')
    feat = csv('data/productores_features_v2.csv')
    st.metric('Productores', len(seg), f"{int(seg['atipico_dbscan'].sum())} atipicos")
    st.bar_chart(seg['segmento'].value_counts().sort_index())
    m = feat.merge(seg[['id_productor', 'segmento']], left_on='ID_PRODUCTOR', right_on='id_productor')
    st.dataframe(m.groupby('segmento')[['VOLUMEN_TOTAL_TN', 'MAX_MERMA_EVENTO', 'PCT_HUM_CRIT']].mean().round(2))
    st.caption('K=2 (base estable vs intensivos con picos) + cuarentena 525. DBSCAN marca atipicos.')
elif frente == 'Fraude':
    st.header('2. Fraude en bascula: score + tipologias')
    top = csv('data/alerta_top.csv')
    sev = st.selectbox('Severidad minima', ['critico', 'alto', 'medio', 'bajo'], index=0)
    orden = {'bajo': 0, 'medio': 1, 'alto': 2, 'critico': 3}
    cols = [c for c in ['id_recepcion', 'fecha_hora', 'id_silo', 'id_contrato', 'peso_neto', 'score', 'tipologia', 'severidad'] if c in top.columns]
    st.dataframe(top[top['severidad'].map(orden) >= orden[sev]][cols].head(500))
    st.bar_chart(top['tipologia'].value_counts())
    st.caption('Severidad por percentil: critico=top 1%. Reglas duras = 100 directo.')
elif frente == 'Regimenes':
    st.header('3. Regimenes IoT + CBOT')
    c1, c2 = st.columns(2)
    with c1:
        ri = csv('data/iot_sensor_regimen.csv')
        st.write('Sensores por regimen dominante')
        st.bar_chart(ri.iloc[:, 1].value_counts().sort_index())
    with c2:
        rm = csv('data/mercado_regimen_hora.csv')
        st.write('Volatilidad por regimen (CBOT: calmo vs volatil)')
        st.bar_chart(rm.groupby('regimen')['volatilidad'].mean())
elif frente == 'Asignacion':
    st.header('4. Asignacion optima a silos')
    a = csv('data/asignacion_optima.csv')
    st.metric('Asignaciones', len(a), f"{a.iloc[:, 1].nunique()} silos usados")
    st.bar_chart(a.iloc[:, 1].value_counts().head(10))
    st.caption('Medido full: baseline 232.0 TN -> optimo 209.1 TN (-9.9%, 22.9 TN/semana). Supuestos: +20% sin control, +10% lleno>90%.')
elif frente == 'Mercado':
    st.header('5-6. Regimenes + cobertura de aceite')
    en = csv('data/precios_enso_mensual.csv')
    st.line_chart(en.set_index(en.columns[0])[[c for c in en.columns if 'SOYA_GRANO' in c.upper()]][:120])
    bt = csv('data/cobertura_aceite_backtest.csv')
    st.write('Backtest walk-forward (cubrir si P(subida)<0.45)')
    st.line_chart(bt[['dlog_soya_aceite']].cumsum().tail(200) if 'dlog_soya_aceite' in bt.columns else bt.iloc[:200, [0]])
    st.caption('Descubrimiento: petroleo->aceite (lags 1-7, p=0.006), Nino->aceite (lags 2-10, p=0.024). +53% vs +40%, vol -28%.')
else:
    st.header('7-8. 360 y jerarquico multidominio')
    st.image('docs/img/dendrograma.png', caption='Dendrograma Ward (48 meses)')
    st.image('docs/img/clustermap.png', caption='Clustermap meses x variables')
    st.caption('Leccion: correlacion -0.65 espuria por 2025 contaminado; limpio |r|<0.3. Coverage contractual ~4% (96% spot).')
