# DGP Bronce: como se generaron los sinteticos (verificado en Snowflake 29/9)

Regla de oro: **modela solo lo que el generador no definio por formula ni sorteo**.

## Formulas exactas (JAMAS predecir: es copiar la formula)
- `VENTA.VALOR_TOTAL_USD = CANTIDAD_TN x PRECIO_UNITARIO_USD` → 54,000/54,000 (100%).
- `RECEPCION peso_neto = PESO_BRUTO - TARA` → 98.9% (resto ruido).
- `PRIMA_BASIS_USO` **NO** es `unit - cbot` (0/54,000): es variable aleatoria propia.

## Sorteos puros (impredecibles por construccion)
- `CALIDAD_GRANO`: PREMIUM/ESTANDAR/DESCUENTO independientes de humedad/impurezas/aceite/proteina
  (medias casi identicas; umbral humedad>14 no cambia proporciones). AUC ~ 0.50 medido.
- `EMBARQUE` retraso 0-15 dias uniforme (~1,100 por dia) + `HORA` recepcion uniforme 6-18h.
- `FECHA_REAL` nula incluso en CANCELADO/PROGRAMADO (no es censura limpia).
- Rango fechas 1990-2099 en todas las tablas (filtrar 2020-2027 para analisis).

## Regularidades utiles (estructura aprovechable)
- IoT: cadencia perfecta 5 min (36,000 lecturas/hora x 24 h), 100 sensores, voltaje constante 3.3.
- `CONTRATO_COMPRA.ESTADO`: CUMPLIDO 4184 / ACTIVO 1519 / CANCELADO 297 (candidato a clasificar,
  verificar antes si deriva de fechas/volumenes).
- `TIPO` transporte 4 clases balanceadas; 3 origenes, 5 destinos/puertos.

## Protocolo anti-humo (5 minutos, antes de cualquier modelo)
1. Muestra 5% + LogReg simple. 2. Si AUC < 0.60 o R2_OOS <= 0: se abandona o se cambia el objetivo.
3. Sospecha de toda metrica "demasiado buena" (AUC > 0.95, R2 > 0.8): buscar la columna formula/fuga.
4. Declarar cuarentenas: 525, contrato 1, ventas 2025, fechas fuera de rango.
