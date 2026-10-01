# Minería Gravetal Bolivia S.A. — 8 frentes (CRISP-DM)

Minería de datos sobre el Lakehouse de Gravetal (Snowflake `AIRBYTE_DATABASE`): segmentación,
fraude, regímenes, asignación óptima, mercado y análisis 360°, con metodología CRISP-DM y
los 11 criterios de evaluación del programa Python.

Cuaderno principal: **`Mineria_Final_Gravetal_CRISP-DM.ipynb`** (modo CSV, sin Snowflake).
Despliegue: **`app.py`** (Streamlit).

## Mapa del repo

| Ruta | Qué es |
|---|---|
| `Mineria_Final_Gravetal_CRISP-DM.ipynb` | Notebook final: 8 frentes + gráficos + cierre |
| `mineria_*.py` | Scripts puros reproducibles (uno por frente) |
| `sql/` | Queries versionados contra Oro/Bronce |
| `data/` | Features y artefactos de resultados (CSVs) |
| `docs/img/` | Dendrograma y clustermap |
| `app.py` | App Streamlit de despliegue (fase 6 CRISP-DM) |
| `diagnostico_bronce.py` | Auditoría DQ de las 44 tablas Bronce |

## Cómo correr

```bash
pip install -r requirements.txt
export SNOWFLAKE_PASSWORD='clave'   # comillas simples por el !
python mineria_productores.py        # o fraude_bascula, regimenes, asignacion_silos, ...
streamlit run app.py                 # despliegue local / túnel Cloudflare
```

## Frente 1 · Segmentos de productores (k-means + DBSCAN)

- **Objetivo:** comprar distinto a cada grupo (precio por riesgo) y focalizar visitas técnicas.
- **Variables (16):** recepciones, volumen total, ticket medio, merma total, humedad media/std/máx,
  % recepciones con humedad>14, impurezas media/std/máx, % impurezas>1, proteína, aceite,
  % descuento, merma máxima de un evento. Log + estandarizado (ninguna domina por escala).
- **Técnicas:** k-means K=2..7 (K por silueta, `n_init` 20–50), DBSCAN (ε por k-distancia) para
  atípicos, ARI split-half como estabilidad.
- **Gráficos y lectura:**
  - *Silueta vs K:* el codo/p máximo elige K. Ojo: silueta 0.96 con tamaños 1896-vs-1 NO es éxito,
    es un outlier aislado (ID 525: 4,381M TN imposibles → cuarentena). Silueta 0.2–0.4 con grupos
    balanceados sí es segmentación real en datos de negocio.
  - *PCA 2D:* nubes separadas = grupos reales; todo mezclado = no hay estructura (no forzar K).
  - *Barras de perfil:* compara medias por segmento. Aquí: humedad máx 45 vs 16.7 y peor evento
    8.95 vs 1.40 TN separan "intensivos con picos" de "base estable" (las medias son idénticas).
- **Resultado:** K=2 + cuarentena 525. DBSCAN: 14 atípicos a revisión individual.

## Frente 2 · Fraude en báscula (score 0–100 + 5 tipologías)

- **Objetivo:** lista priorizada de auditoría con volumen predecible (no flag binario).
- **Variables (por pesaje, 7 cruces Bronce):** peso/tara/hora, z robustos por silo-mes, rendimiento
  TN/ha vs predio, sobre-entrega vs contrato, laboratorio ausente, madrugada/domingo, Benford
  por silo-mes (chi-cuadrado), score IsolationForest.
- **Técnicas:** reglas duras (100 directo: física imposible, triple-hit documental), ensemble
  ponderado, severidad por **percentil** (crítico = top 1%, no umbrales fijos).
- **Gráficos y lectura:**
  - *Barras por tipología* (FISICA/DOCUMENTAL/RENDIMIENTO/ESTADISTICA/HORARIO): dice *qué tipo*
    de problema domina; cada una va a otro responsable (planta vs comercial).
  - *Histograma de score:* debe verse cola larga a la derecha (pocos muy sospechosos). Si todo
    se amontona arriba, los pesos están mal calibrados.
  - *Top silos/contratos:* el contrato 1 concentrando basura es hallazgo, no error del modelo.
- **Resultado:** tipologías repartidas, severidad bajo/medio/alto/crítico balanceada, top-5000 auditable.

## Frente 3 · Regímenes IoT + CBOT

- **Objetivo:** semáforo operativo (ventilar/mantener por régimen) y de trading (cubrir solo en calmo).
- **Variables IoT:** temperatura, humedad, precipitación, señal, hora (voltaje descartado: constante 3.3).
  **CBOT:** volatilidad, volumen y rango por hora-sesión (15,080 ticks → 3,640 horas).
- **Técnicas:** MiniBatchKMeans (864k filas), DBSCAN en muestra (atípicos de red), k-means en sesiones.
- **Gráficos y lectura:**
  - *Perfiles medios por régimen:* cada fila debe leerse como un "clima" con nombre (madrugada seca/
    húmeda, tarde buena/mala señal, lluvia). Si dos filas son indistinguibles, sobra un K.
  - *Ranking de sensores calientes:* por dónde empezar la ronda (SNS-036, 072…).
  - *Volatilidad por régimen:* calmo 0.0006 vs volátil 0.0024 (4×) con silueta 0.519 = separación fuerte.
- **Resultado:** IoT K=5 operativos + 2.7% atípicos; CBOT K=2 (3,416 calmas vs 224 volátiles).

## Frente 4 · Asignación óptima a silos (PuLP/CBC)

- **Objetivo:** cada camión al silo que minimice merma esperada.
- **Variables:** merma base (humedad/impurezas sobre 14%/1%) × factor silo (sin control +20%,
  lleno>90% +10%), capacidad nominal, 1,637 lotes × 85 silos.
- **Gráficos y lectura:** barras real vs óptimo + top silos cargados. Medido full: 232.0 → 209.1 TN
  (**−9.9%**, 22.9 TN/semana ≈ 1,190 TN/año) con estado `Optimal`.
- **Supuestos a declarar:** factores estimados (sensibilidad pendiente) e inventario inconsistente
  con capacidad (sintético) → se planifica sobre capacidad nominal.

## Frente 5 · Regímenes del mercado sojero (Kaggle 1960–2025)

- **Objetivo:** contexto de ciclo para plan anual y cobertura.
- **Variables:** retornos mensuales soya/aceite/harina, petróleo, maíz, anomalía Niño3.4 (792 meses).
- **Técnicas:** ADF (estacionariedad), k-means, ANOVA, DBSCAN, Granger, PCA.
- **Gráficos y lectura:**
  - *Retorno acumulado coloreado:* se ven los superciclos a simple vista.
  - *PCA:* dos nubes = alcista/bajista (F=453, p=8.5e-80).
  - *Crisis en rojo:* DBSCAN pesca 1973, crash 2008 (−27%), COVID y Niños 97/15 sin saber historia.
- **Resultado:** K=2 (+4.4% vs −2.6% mensual). Granger Niño→grano negativo (hallazgo honesto).

## Frente 6 · Cobertura de aceite (descubrimiento + backtest)

- **Descubrimiento (Granger lags 1–12):** el grano no responde a nada; el **aceite** sí —
  petróleo→aceite (lags 1–7, p=0.006, canal biodiésel), Niño→aceite (lags 2–10, p=0.024).
- **Aplicación:** regla walk-forward (cubrir si P(subida)<0.45, sin mirar futuro).
- **Gráficos y lectura:**
  - *Equity buy&hold vs cobertura:* +53% vs +40%, volatilidad −28%.
  - *Confusión:* hit-rate 0.50 (casi azar). La lectura correcta: el valor es **esquivar caídas
    grandes**, no adivinar dirección. Reportar costos de transacción como trabajo futuro.

## Frente 7 · 360° (cobertura, drivers, basis)

- **Variables:** cumplimiento contractual, 8 drivers GBR, basis por segmento.
- **Gráficos y lectura:**
  - *Histograma de cobertura:* casi todo en ~0 → contratos cubren ~4% (96% spot). De ahí sale el
    programa de forwards, el hallazgo de más plata.
  - *Drivers (GBR R²~0.80):* humedad 3× impurezas; el mes pesa igual que impurezas → secado +
    timing de cosecha, no laboratorio. Proteína/aceite/temp/hora ≈ 0.
  - *Basis por segmento + ANOVA:* refinería −25.6 vs trader −28.5 USD/TN → renegociar traders.

## Frente 8 · Jerárquico multidominio

- **Variables:** 12 mensuales 2021–24 (mercado × operación × comercial). IoT excluido (solo trae 2026).
- **Gráficos y lectura:**
  - *Dendrograma:* ramas que se juntan abajo = meses gemelos (picos de cosecha, shocks). Altura =
    disimilitud. Método por cofenética (average 0.72).
  - *Clustermap:* bloques de color = qué variables se mueven juntas y cuándo.
  - *Lección estrella:* sin limpiar, r=−0.65 espuria (2025 con 800 ventas imposibles); limpio,
    |r|<0.3. Reportar el nulo también es minería.

## Diagnóstico Bronce (44 tablas)

Sólidos: recepción 1.55M, telemetría 864k, cotizaciones 15k, venta/envío/control. Descartar:
`CALIBRACION_MANTENIMIENTO` vacía, `OBSERVACIONES` y `_AB_CDC_DELETED_AT` 100% nulas, tablas de
2–10 filas (solo etiquetas). Cuidado: `EMAIL` 39%, `SISTEMA_RIEGO` 21%, `FECHA_REAL` 10%,
`CAMPAÑA` con Ñ (citar entre comillas), `DUP_CLAVE` en asignaciones/producción = grano 1→N real.

## Limitaciones honestas

Datos sintéticos con artefactos conocidos (525, contrato 1, 2025 comercial, inventarios,
fechas 1990/2099). Cada frente documenta su cuarentena. Muestreos (`TABLESAMPLE`) cambian
decimales entre corridas: lo direccional es estable, los decimales no.
