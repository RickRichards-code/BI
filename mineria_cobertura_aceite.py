#!/usr/bin/env python3
"""Descubrimiento: el ACEITE (no el grano) responde a petroleo (biodiesel) y El Nino.
Granger lags 1-12 + senal direccional walk-forward + backtest de cobertura.
IMPORTANCIA: regla 후보 para mesa de trading (cubrir aceite cuando la senal dispara).
Uso:  python mineria_cobertura_aceite.py
Requiere: statsmodels, scikit-learn. Lee data/precios_enso_mensual.csv (repo).
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
SEED = 42
def main():
    df = pd.read_csv('data/precios_enso_mensual.csv')
    df.columns = df.columns.str.lower()
    df['mes'] = pd.to_datetime(df['mes'], errors='coerce')
    for c in ['dlog_soya_aceite', 'dlog_petroleo', 'nino34_anomalia_c', 'dlog_soya_grano']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.sort_values('mes').dropna(subset=['dlog_soya_aceite', 'dlog_petroleo', 'nino34_anomalia_c']).copy()
    # 1. Matriz Granger lags 1-12 (descubrimiento)
    from statsmodels.tsa.stattools import grangercausalitytests
    import io, contextlib
    pairs = [('dlog_petroleo', 'dlog_soya_aceite'), ('nino34_anomalia_c', 'dlog_soya_aceite'),
             ('dlog_petroleo', 'dlog_soya_grano'), ('nino34_anomalia_c', 'dlog_soya_grano')]
    print('=== Granger min-p y lags significativos ===')
    for cause, eff in pairs:
        g = df[[eff, cause]].dropna()
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                out = grangercausalitytests(g, maxlag=12, verbose=False)
            except TypeError:
                out = grangercausalitytests(g, maxlag=12)
        ps = {lag: out[lag][0]['ssr_ftest'][1] for lag in range(1, 13)}
        print(f'{cause} -> {eff}: min p={min(ps.values()):.4f} lags={[l for l, p in ps.items() if p < 0.05]}')
    # 2. Senal direccional walk-forward: sube el aceite el proximo mes?
    d = df.copy()
    d['y'] = (d['dlog_soya_aceite'].shift(-1) > 0).astype(int)
    for L in (1, 2):
        d[f'pet_l{L}'] = d['dlog_petroleo'].shift(L)
    d['nino_l2'] = d['nino34_anomalia_c'].shift(2)
    feats = ['pet_l1', 'pet_l2', 'nino_l2']
    d = d.dropna(subset=feats + ['y']).copy()
    X, y = d[feats].values, d['y'].values
    # expanding window desde 50% (sin mirar futuro)
    t0 = len(d) // 2
    preds = np.zeros(len(d))
    for t in range(t0, len(d)):
        m = LogisticRegression(max_iter=500).fit(X[:t], y[:t])
        preds[t] = m.predict_proba(X[t:t + 1])[0, 1]
    te = slice(t0, len(d))
    hit = accuracy_score(y[te], (preds[te] > 0.5).astype(int))
    auc = roc_auc_score(y[te], preds[te])
    print(f'=== Walk-forward (n={len(y[te])}): hit-rate={hit:.3f} AUC={auc:.3f} (azar=0.50) ===')
    # 3. Backtest cobertura: cubierto (=0 variacion) cuando senal>0.55, si no expuesto
    r = d['dlog_soya_aceite'].values[te]
    # Regla honesta: cubrirse (=congelar precio, ret 0) cuando se pronostica CAIDA: protege ingreso.
    strat = np.where(preds[te] < 0.45, 0.0, r)
    bh = np.exp(np.log1p(r).cumsum())
    st = np.exp(np.log1p(strat).cumsum())
    vol_bh, vol_st = r.std(), strat.std()
    print(f'Buy&hold: ret acum={bh[-1] - 1:+.2%} vol={vol_bh:.4f} | Cobertura: ret={st[-1] - 1:+.2%} vol={vol_st:.4f}')
    print(f'Reduccion de volatilidad: {(1 - vol_st / vol_bh) * 100:.1f}%')
    d.assign(pred=preds).to_csv('data/cobertura_aceite_backtest.csv', index=False, encoding='utf-8')
    print('Guardado data/cobertura_aceite_backtest.csv')
if __name__ == '__main__':
    main()
