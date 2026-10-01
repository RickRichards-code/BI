# Minería Gravetal Bolivia S.A. — guía completa (para cualquier persona)

Proyecto de **minería de datos** sobre el Lakehouse de Gravetal (empresa que acopia soya,
la procesa y la exporta). Todo verificado contra Snowflake (`AIRBYTE_DATABASE`) y replicado
en dos máquinas. Si solo lees algo, lee [Lo esencial en 2 minutos](#lo-esencial-en-2-minutos).

## Lo esencial en 2 minutos

**Pregunta del proyecto:** con los datos que ya tiene Gravetal, ¿qué decisiones se pueden
mejorar con minería de datos? **Respuesta en 8 hallazgos:**

| # | Hallazgo (una línea) | Plata |
|---|---|---|
| 1 | Hay 2 tipos de productores: base estable e intensivos con picos de humedad | Precio y visitas distintos |
| 2 | Cada pesaje recibe score 0–100 + tipología (física, documental, rendimiento, estadística, horaria) | Auditoría priorizada |
| 3 | El clima de silos tiene 5 regímenes; el mercado 2 (calmo/volátil 4×) | Ventilar y cubrir a tiempo |
| 4 | Asignar cada camión al mejor silo ahorra 9.9% de merma (22.9 TN/semana) | ~1,190 TN/año |
| 5 | El mercado sojero tiene régimen alcista/bajista + 175 meses de crisis reales | Plan anual |
| 6 | El **aceite** (no el grano) responde al petróleo y El Niño → regla de cobertura +53% vs +40% | Proteger ingresos |
| 7 | Los contratos cubren ~4% (96% a spot); la humedad pesa 3× más que impurezas; refinería paga +3 USD/TN | Forwards, secado, renegociar |
| 8 | Sin limpiar, una correlación −0.65 era basura (2025 contaminado); limpio no hay cruces fuertes | Rigor = también hallazgo |

**Archivos para empezar:** cuaderno `Mineria_Final_Gravetal_CRISP-DM.ipynb` (relato + gráficos),
`app.py` (demo viva con Streamlit), scripts `mineria_*.py` (cómputo reproducible).

## Glosario (sin jerga)

- **Cluster/segmento:** grupo de registros parecidos entre sí. *Ejemplo:* productores que entregan
  mucho volumen con picos de humedad van juntos aunque nunca te lo dijeron.
- **k-means:** algoritmo que parte los datos en K grupos buscando que cada punto quede cerca del
  centro de su grupo. Hay que decirle K; se elige probando 2..7 y mirando la silueta.
- **Silueta (0 a 1):** qué tan bien separados están los grupos. ~0.2–0.4 = normal en negocio real.
  **Trampa famosa:** 0.96 con grupos de 1896-vs-1 NO es éxito, es un outlier aislado (nos pasó con
  el productor 525: 4,381 millones de TN imposibles → cuarentena).
- **DBSCAN:** encuentra grupos de cualquier forma y marca **atípicos** (puntos en zonas vacías).
  No pide K. Ideal para "qué es raro aquí" (sensores fallando, meses de crisis).
- **IsolationForest:** aísla lo anómalo: lo raro se separa con pocos cortes. Da un score de rareza.
- **Benford:** en datos reales, el primer dígito de los números sigue una distribución conocida
  (el 1 aparece ~30%). Números inventados no la siguen → test chi-cuadrado los delata.
- **PCA:** aplasta muchas variables a 2 ejes para dibujar la nube de puntos. Si se ven grupos
  separados a ojo, hay estructura; si es una mancha, no la hay.
- **Dendrograma:** árbol que muestra qué se parece a qué: ramas que se juntan abajo = gemelos.
- **Clustermap:** tabla pintada por colores + dendrogramas: revela bloques (qué variables se
  mueven juntas y cuándo).
- **Granger:** test estadístico de "X anticipa a Y" (con rezagos 1–12 meses). p<0.05 = hay señal.
  Ojo: anticipar ≠ causar en sentido físico, pero para trading basta.
- **ADF:** verifica que una serie no tenga tendencia explosiva (requisito antes de modelarla).
- **ANOVA F:** mide si los grupos difieren de verdad en una variable (F grande + p≈0 = sí).
- **ARI (0 a 1):** estabilidad: ¿sale lo mismo si corro dos veces? ≥0.60 = estable.
- **Walk-forward:** entrenar solo con pasado y probar con futuro, avanzando mes a mes (prohibido
  mirar el futuro = leakage).
- **Backtest:** simular la regla sobre historia y medir plata y volatilidad resultantes.
- **PuLP/CBC:** optimizador que asigna cada camión a un silo respetando capacidades.
- **Severidad por percentil:** crítico = top 1% del score. Así el volumen a auditar es fijo y
  predecible, no "los que salgan".

## Cómo correr todo

```bash
pip install -r requirements.txt
export SNOWFLAKE_PASSWORD='clave'   # comillas simples (suele terminar en !)
python diagnostico_bronce.py        # auditoría de las 44 tablas
python mineria_productores.py       # segmentos (+DBSCAN)
python mineria_fraude_bascula.py    # score + tipologías (1.55M; SAMPLE_PCT=1 para probar)
python mineria_regimenes.py         # IoT + CBOT
python mineria_asignacion_silos.py  # PuLP (DIAS=7; tarda minutos al 100%)
python mineria_regimen_mercado.py   # Kaggle ENSO (pip install statsmodels)
python mineria_cobertura_aceite.py  # Granger + backtest (lee data/*.csv, rapidísimo)
python mineria_360_productores.py   # compliance + drivers + basis
python mineria_jerarquico.py        # dendrograma + clustermap (lee matriz CSV)
streamlit run app.py                # despliegue (o túnel Cloudflare para mostrarlo)
```

## Frente por frente (detalle)

### F1 · Segmentos de productores — *¿a quién le compro cómo?*
- **Datos:** 1,897 productores agregados de 1.55M recepciones (volumen, frecuencia, ticket,
  merma, humedad media/std/máx, % eventos críticos, impurezas igual, proteína, aceite,
  % descuento, peor evento). Log + estandarizado para que ninguna variable pese más por escala.
- **Técnica:** k-means K=2..7 (K por silueta) + DBSCAN (ε por k-distancia) + ARI split-half.
- **Gráficos:**
  - *Silueta vs K:* elige el pico. Si el pico viene con un grupo de 1 solo miembro, es un
    outlier, no un segmento: cuarentena y repite.
  - *PCA 2D:* dos nubes = dos negocios distintos; una mancha = no fuerces grupos.
  - *Barras de perfil:* lee qué distingue a cada grupo. Aquí: humedad máx 45 vs 16.7 y peor
    evento 8.95 vs 1.40 TN (las medias son idénticas: el riesgo está en los picos, no en
    el promedio).
- **Acción:** precio diferencial por grupo + visitas solo a intensivos/atípicos.

### F2 · Fraude en báscula — *¿qué pesaje audito primero?*
- **Datos:** 1.55M pesajes crudos + 7 cruces (contrato, predio, laboratorio, inventario, planta,
  venta, remisión). Nada de Oro cocido: se necesita la precisión original.
- **Técnica:** reglas duras (100 directo: física imposible, triple-hit documental) + ensemble
  (Benford por silo-mes, IsolationForest, rendimiento TN/ha, sobre-entrega, horario) con
  severidad por percentil.
- **Gráficos:**
  - *Barras por tipología:* FISICA→planta, DOCUMENTAL→comercial, RENDIMIENTO→campo,
    ESTADISTICA→muestreo, HORARIO→turnos. Si una barra se come todo, recalibra pesos.
  - *Histograma de score:* sano = montaña a la izquierda + cola larga a la derecha.
  - *Top silos/contratos:* un concentrador (contrato 1) es hallazgo, no bug.
- **Acción:** auditar el top-k por tipología cada mes.

### F3 · Regímenes — *¿en qué modo estamos?*
- **IoT (864k lecturas):** temp, humedad, precipitación, señal, hora (voltaje descartado: fijo en
  3.3). MiniBatchKMeans (k-means clásico no escala) + DBSCAN en muestra.
- **CBOT (15,080 ticks → 3,640 horas):** volatilidad, volumen y rango por hora.
- **Gráficos:**
  - *Perfiles medios:* cada fila debe leerse como un clima con nombre (madrugada seca/húmeda,
    tarde buena/mala señal, lluvia). Dos filas iguales = sobra un K.
  - *Ranking de sensores:* por dónde empezar la ronda de mantenimiento.
  - *Volatilidad por régimen:* calmo 0.0006 vs volátil 0.0024 (4×), silueta 0.519 = separación
    fuerte y accionable.

### F4 · Asignación a silos — *¿dónde va cada camión?*
- **Modelo:** merma = peso×(humedad/impurezas sobre umbral) × factor silo (sin control +20%,
  lleno>90% +10%), minimizada con PuLP/CBC bajo capacidad. 1,637×85 en el full.
- **Gráficos:** barras real vs óptimo + top silos cargados. Medido: 232.0→209.1 TN (−9.9%).
- **Declarar siempre:** factores estimados (falta sensibilidad) e inventario inconsistente con
  capacidad (sintético) → capacidad nominal.

### F5 · Mercado sojero Kaggle — *¿en qué ciclo estamos?*
- **Datos:** 792 meses reales 1960–2025 (soya, maíz, petróleo, fertilizantes, Niño3.4 + rezagos).
- **Técnicas:** ADF, k-means, ANOVA, DBSCAN, Granger, PCA.
- **Gráficos:**
  - *Retorno acumulado por régimen:* los superciclos se ven sin estadística.
  - *PCA:* dos nubes (alcista/bajista, F=453).
  - *Crisis en rojo:* 1973, 2008 (−27%), COVID, Niños 97/15 hallados sin saber historia.

### F6 · Cobertura de aceite — *¿cubro este mes?*
- **Descubrimiento:** petróleo→aceite (lags 1–7, p=0.006, biodiésel) y Niño→aceite (lags 2–10,
  p=0.024). El grano no responde a nada: el negocio estaba en el aceite.
- **Regla walk-forward:** cubrir (=congelar) si P(subida)<0.45. +53% vs +40%, volatilidad −28%.
- **Gráficos:** equity de ambas curvas + confusión. **Lectura honesta:** hit-rate 0.50 (casi azar);
  el valor es esquivar caídas grandes, no adivinar. Costos de transacción = trabajo futuro.

### F7 · 360° — *¿dónde está la plata estructural?*
- Cobertura contractual ~4% (histograma pegado a 0) → programa de forwards (96% a spot hoy).
- Drivers GBR (R²~0.80): humedad 3× impurezas; el mes pesa igual que impurezas → secado + timing,
  no laboratorio. Proteína/aceite/temp/hora ≈ 0.
- Basis ANOVA: refinería −25.6 vs trader −28.5 → renegociar +3 USD/TN.

### F8 · Jerárquico multidominio — *¿qué se mueve junto?*
- 60 meses × 12 variables (mercado×operación×comercial; IoT solo trae 2026, fuera).
- **Dendrograma** (average, cofenética 0.72): ramas bajas = meses gemelos (cosechas, shocks).
- **Clustermap:** bloques de color = variables que se mueven juntas.
- **Lección:** sin limpiar, r=−0.65 espuria por 800 ventas imposibles de 2025; limpio, |r|<0.3.
  Reportar el nulo también es minería.

## Diagnóstico Bronce (resumen)

Sólidos: recepción 1.55M, telemetría 864k, cotizaciones 15k, venta/envío/control. Descartar:
`CALIBRACION_MANTENIMIENTO` vacía, `OBSERVACIONES`/`_AB_CDC_DELETED_AT` 100% nulas, tablas de
2–10 filas (etiquetas). Cuidado: `EMAIL` 39%, `SISTEMA_RIEGO` 21%, `FECHA_REAL` 10%,
`CAMPAÑA` con Ñ (entre comillas siempre), `DUP_CLAVE` en asignaciones/producción = grano 1→N real.

## Limitaciones (léeme antes de defender)

Sintéticos con artefactos conocidos y cuarentenados (525, contrato 1, ventas 2025, inventarios,
fechas 1990/2099). `TABLESAMPLE` cambia decimales entre corridas: lo direccional es estable,
los decimales no. Reporta rangos, no dígitos.
