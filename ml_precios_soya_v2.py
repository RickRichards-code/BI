#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROYECTO 1 v2 | Pronostico de precios de soya (grano / aceite / harina) y margen de crush
========================================================================================
v2 conserva TODA la complejidad del original y corrige: (1) random_state en todos los
modelos estocasticos (replicabilidad entre maquinas); (2) class_weight='balanced' en
clasificadores (LATERAL domina 50-70%); (3) sin ARIMA: 100% ML (Naive/Media solo referencias);
(4) intervalos conformales calibrados con el
modelo REAJUSTADO sobre validacion (antes usaban residuos del modelo solo-train);
(5) Diebold-Mariano con correccion de Holm por familia (objetivo,h); (6) calibracion
isotonic/sigmoid del campeon clasificador con Brier/LogLoss antes/despues;
(7) versiones de librerias impresas al inicio para auditar replicabilidad.
"""
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (ElasticNetCV, LassoCV, LogisticRegression, RidgeCV)
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, r2_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.calibration import CalibratedClassifierCV

# ----------------------------------------------------------------------------- CONFIG
SEED = 42
np.random.seed(SEED)
TABLA = os.getenv('TABLA_PRECIOS', 'ARQUI1.CAPA_BRONCE.PRECIOS_ENSO_MENSUAL')
QUICK = os.getenv('QUICK', '0') == '1'
HORIZONTES = [int(x) for x in os.getenv('HORIZONTES', '3' if QUICK else '1,3,6').split(',')]
TARGETS = [x.strip().lower() for x in os.getenv(
    'TARGETS', 'soya_grano' if QUICK else 'soya_grano,soya_aceite,soya_harina,crush_spread').split(',')]
OUT_DIR = os.getenv('OUT_DIR', 'resultados_soya')
ESCRIBIR = os.getenv('ESCRIBIR', '0') == '1'
FRAC_TRAIN, FRAC_VAL = 0.70, 0.15
ALPHA = 0.20
W_HARINA, W_ACEITE = 0.80, 0.18
BASE = ['soya_grano', 'soya_aceite', 'soya_harina', 'maiz', 'urea', 'dap', 'potasio', 'petroleo']
ENSO = 'nino34_anomalia_c'
CLASES = {0: 'BAJA', 1: 'LATERAL', 2: 'ALZA'}


def versiones():
    mods = {}
    for nom in ('numpy', 'pandas', 'scipy', 'sklearn', 'statsmodels'):
        try:
            mods[nom] = __import__('importlib').import_module(nom).__version__
        except Exception:
            mods[nom] = 'no-instalado'
    return mods


class Tee:
    """Duplica stdout a un archivo de log."""
    def __init__(self, path):
        self.f = open(path, 'w', encoding='utf-8')
        self.o = sys.stdout

    def write(self, s):
        self.o.write(s)
        self.f.write(s)

    def flush(self):
        self.o.flush()
        self.f.flush()


def banner(t, ch='='):
    print('\n' + ch * 100 + f'\n{t}\n' + ch * 100)


def tabla(df, cols=None, nd=4):
    if df is None or len(df) == 0:
        return '(vacio)'
    d = df if cols is None else df[[c for c in cols if c in df.columns]]
    return d.to_string(index=False, float_format=lambda x: f'{x:,.{nd}f}')


def holm(pvals, alpha=0.05):
    """Holm-Bonferroni: devuelve mascara de rechazos e indices ordenados."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    rej = np.zeros(len(p), bool)
    order = np.argsort(np.where(ok, p, np.inf))
    m = int(ok.sum())
    for rank, j in enumerate(order[:m]):
        if p[j] <= alpha / (m - rank):
            rej[j] = True
        else:
            break
    return rej

# ----------------------------------------------------------------------------- DATOS
class Backend:
    def __init__(self):
        self.session, self.conn, self.mode = None, None, None
        if os.getenv('DATA_CSV'):
            self.mode = 'csv'
            print(f'Origen: CSV local {os.getenv("DATA_CSV")}')
            return
        try:
            from snowflake.snowpark.context import get_active_session
            self.session = get_active_session()
            self.mode = 'snowpark'
            print('Origen: sesion Snowflake (Snowpark), sin credenciales')
        except Exception:
            import snowflake.connector
            pwd = os.getenv('SNOWFLAKE_PASSWORD', '').strip().strip('\'"')
            if not pwd:
                raise SystemExit("Sin password: export SNOWFLAKE_PASSWORD='clave' (o usa DATA_CSV=archivo.csv)")
            kw = dict(user=os.getenv('SNOWFLAKE_USER', 'ENRIQUE'), password=pwd,
                      account=os.getenv('SNOWFLAKE_ACCOUNT', 'WMNAMCT-LN60692'),
                      database=os.getenv('SNOWFLAKE_DB', 'ARQUI1'),
                      schema=os.getenv('SNOWFLAKE_SCHEMA', 'CAPA_BRONCE'),
                      warehouse=os.getenv('SNOWFLAKE_WH', 'COMPUTE_WH'), login_timeout=60)
            if os.getenv('SNOWFLAKE_ROLE'):
                kw['role'] = os.getenv('SNOWFLAKE_ROLE')
            self.conn = snowflake.connector.connect(**kw)
            self.mode = 'connector'
            print('Origen: connector local')

    def cargar_precios(self):
        if self.mode == 'csv':
            return pd.read_csv(os.getenv('DATA_CSV'))
        q = f"""SELECT * FROM {TABLA}
                QUALIFY ROW_NUMBER() OVER (PARTITION BY MES ORDER BY _AIRBYTE_EXTRACTED_AT DESC) = 1
                ORDER BY MES"""
        if self.mode == 'snowpark':
            return self.session.sql(q).to_pandas()
        cur = self.conn.cursor()
        cur.execute(q)
        return pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])

    def escribir(self, df, nombre):
        if df is None or len(df) == 0:
            return
        out = df.copy()
        out.columns = [str(c).upper() for c in out.columns]
        try:
            if self.mode == 'snowpark':
                self.session.write_pandas(out, nombre, auto_create_table=True, overwrite=True)
            elif self.mode == 'connector':
                from snowflake.connector.pandas_tools import write_pandas
                write_pandas(self.conn, out, nombre, auto_create_table=True, overwrite=True)
            else:
                return
            print(f'  -> tabla {nombre} escrita en Snowflake')
        except Exception as ex:
            print(f'  [!] no se pudo escribir {nombre}: {ex}')


def preparar(df):
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    df['mes'] = pd.to_datetime(df['mes']).dt.to_period('M').dt.to_timestamp()
    orden = ['mes'] + (['_airbyte_extracted_at'] if '_airbyte_extracted_at' in df.columns else [])
    n0 = len(df)
    df = df.sort_values(orden).drop_duplicates('mes', keep='last')
    print(f'Filas bronce: {n0} -> meses unicos: {len(df)}')
    cols = [c for c in BASE + [ENSO] if c in df.columns]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'split' in df.columns:
        print('\nColumna SPLIT de la tabla (informativa; este script usa corte cronologico 70/15/15):')
        print(df.groupby('split')['mes'].agg(['min', 'max', 'count']).to_string())
    d = df.set_index('mes')[cols].asfreq('MS')
    print(f'Meses faltantes en el calendario continuo: {int(d[BASE[:3]].isna().any(axis=1).sum())}')
    d = d.ffill(limit=2)
    precios = [c for c in BASE[:3] if c in d.columns]
    if len(precios) < 3:
        raise SystemExit('Faltan columnas soya_grano / soya_aceite / soya_harina')
    ok = d[precios].notna().all(axis=1)
    d = d.loc[ok.idxmax(): ok[::-1].idxmax()]
    print(f'Serie final: {len(d)} meses  {d.index.min().date()} -> {d.index.max().date()}')
    print('\nResumen de niveles:')
    print(d.describe().T[['count', 'mean', 'std', 'min', 'max']].round(2).to_string())
    return d

# ----------------------------------------------------------------------------- FEATURES
def construir_features(d):
    F = {}
    series = [s for s in BASE if s in d.columns and d[s].notna().sum() > 50]
    L = np.log(d[series].where(d[series] > 0))
    for s in series:
        lp = L[s]
        r1 = lp.diff()
        F[f'{s}__ret_1m'] = r1
        F[f'{s}__ret_3m'] = lp.diff(3)
        F[f'{s}__ret_6m'] = lp.diff(6)
        F[f'{s}__ret_12m'] = lp.diff(12)
        for k in (1, 2, 3):
            F[f'{s}__ret_1m_lag{k}'] = r1.shift(k)
        F[f'{s}__vol_6m'] = r1.rolling(6).std()
        F[f'{s}__vol_12m'] = r1.rolling(12).std()
        F[f'{s}__gap_ma12'] = lp - lp.rolling(12).mean()
        F[f'{s}__z_24m'] = (lp - lp.rolling(24).mean()) / lp.rolling(24).std()
    pares = [('soya_aceite', 'soya_grano', 'aceite_grano'), ('soya_harina', 'soya_grano', 'harina_grano'),
             ('soya_grano', 'maiz', 'grano_maiz'), ('soya_grano', 'urea', 'grano_urea'),
             ('soya_grano', 'dap', 'grano_dap'), ('petroleo', 'soya_grano', 'petroleo_grano'),
             ('soya_aceite', 'petroleo', 'aceite_petroleo')]
    for a, b, nom in pares:
        if a in L and b in L:
            r = L[a] - L[b]
            F[f'rat__{nom}'] = r
            F[f'rat__{nom}_d3'] = r.diff(3)
            F[f'rat__{nom}_z24'] = (r - r.rolling(24).mean()) / r.rolling(24).std()
    sp = W_HARINA * d['soya_harina'] + W_ACEITE * d['soya_aceite'] - d['soya_grano']
    F['crush__spread'] = sp
    F['crush__spread_pct'] = sp / d['soya_grano']
    F['crush__d3'] = sp.diff(3)
    F['crush__z24'] = (sp - sp.rolling(24).mean()) / sp.rolling(24).std()
    if ENSO in d.columns and d[ENSO].notna().sum() > 50:
        n = d[ENSO]
        F['enso__anom'] = n
        for k in (1, 3, 6, 9):
            F[f'enso__anom_lag{k}'] = n.shift(k)
        F['enso__media3'] = n.rolling(3).mean()
        F['enso__media6'] = n.rolling(6).mean()
        F['enso__d3'] = n.diff(3)
        F['enso__fase_nino'] = (n >= 0.5).astype(float).where(n.notna())
        F['enso__fase_nina'] = (n <= -0.5).astype(float).where(n.notna())
    m = d.index.month
    F['est__mes_sin'] = pd.Series(np.sin(2 * np.pi * m / 12), index=d.index)
    F['est__mes_cos'] = pd.Series(np.cos(2 * np.pi * m / 12), index=d.index)
    X = pd.DataFrame(F, index=d.index).replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.isna().mean() < 0.25]
    return X


def serie_objetivo(d, nombre):
    if nombre == 'crush_spread':
        return W_HARINA * d['soya_harina'] + W_ACEITE * d['soya_aceite'] - d['soya_grano'], 'diff'
    return d[nombre], 'logret'


def cambio_futuro(lvl, kind, h):
    v = lvl.values.astype(float)
    out = np.full(len(v), np.nan)
    with np.errstate(all='ignore'):
        out[:len(v) - h] = np.log(v[h:] / v[:-h]) if kind == 'logret' else v[h:] - v[:-h]
    out[~np.isfinite(out)] = np.nan
    return out


def particiones(valid, h):
    n = len(valid)
    a, b = int(n * FRAC_TRAIN), int(n * (FRAC_TRAIN + FRAC_VAL))
    return valid[:a], valid[a + h:b], valid[b + h:], valid[:b]


# ----------------------------------------------------------------------------- PCA
def analisis_pca(X, cols, y3m):
    banner('ANALISIS PCA (ajustado solo con el bloque de entrenamiento)')
    Z = StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(X))
    pca = PCA(random_state=SEED).fit(Z)
    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)
    ncomp = {f'{int(t * 100)}%': int(np.searchsorted(cum, t) + 1) for t in (0.80, 0.90, 0.95, 0.99)}
    print(f'Variables: {len(cols)} | filas train: {len(Z)} | n. de condicion X estandarizada: {np.linalg.cond(Z):,.1f}')
    print('Varianza explicada PC1..PC10 (%):', np.round(ev[:10] * 100, 2).tolist())
    print('Varianza acumulada PC1..PC10 (%):', np.round(cum[:10] * 100, 2).tolist())
    print('Componentes necesarios:', ncomp)
    load = pd.DataFrame(pca.components_[:5].T, index=cols, columns=[f'PC{i + 1}' for i in range(5)])
    for pc in ['PC1', 'PC2', 'PC3']:
        top = load[pc].reindex(load[pc].abs().sort_values(ascending=False).index)[:6]
        print(f'Top cargas {pc}: ' + ' | '.join(f'{k}={v:+.2f}' for k, v in top.items()))
    grupos = pd.Series([c.split('__')[0] for c in cols], index=cols)
    contrib = ((load ** 2).groupby(grupos).sum() * 100).round(1)
    print('\nContribucion por grupo de variables a cada PC (%):')
    print(contrib.to_string())
    sc = pca.transform(Z)[:, :5]
    ok = ~np.isnan(y3m[:len(sc)])
    corr = {f'PC{i + 1}': round(float(stats.spearmanr(sc[ok, i], y3m[:len(sc)][ok])[0]), 3) for i in range(5)}
    print('\nSpearman PC vs retorno futuro 3m del grano:', corr)
    return dict(ncomp=ncomp, ev=ev[:5], cond=float(np.linalg.cond(Z)), contrib=contrib, corr=corr)


# ----------------------------------------------------------------------------- MODELOS
class PLSFlat(PLSRegression):
    def predict(self, X, copy=True):
        return np.ravel(super().predict(X, copy=copy))


def modelos_reg():
    ts = TimeSeriesSplit(n_splits=4)
    imp = lambda: SimpleImputer(strategy='median')
    pca = lambda v: PCA(n_components=v, svd_solver='full', random_state=SEED)
    M = {}
    M['Media_hist'] = DummyRegressor(strategy='mean')
    M['Ridge'] = make_pipeline(imp(), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 25), cv=ts))
    M['PCA_Ridge'] = make_pipeline(imp(), StandardScaler(), pca(0.90), RidgeCV(alphas=np.logspace(-2, 4, 25), cv=ts))
    M['PLS'] = GridSearchCV(make_pipeline(imp(), StandardScaler(), PLSFlat(scale=False)),
                            {'plsflat__n_components': [1, 2, 3, 4, 6, 8]}, cv=ts,
                            scoring='neg_mean_absolute_error')
    M['RF'] = make_pipeline(imp(), RandomForestRegressor(n_estimators=300, min_samples_leaf=8, max_features=0.33,
                                                         n_jobs=-1, random_state=SEED))
    M['HGB'] = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.04, max_depth=3,
                                             min_samples_leaf=15, l2_regularization=1.0, random_state=SEED)
    if QUICK:
        return M
    M['Lasso'] = make_pipeline(imp(), StandardScaler(), LassoCV(cv=ts, alphas=np.logspace(-5, -0.5, 40), max_iter=50000, random_state=SEED))
    M['ElasticNet'] = make_pipeline(imp(), StandardScaler(),
                                    ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8, 0.95], cv=ts, alphas=np.logspace(-5, -0.5, 30),
                                                 max_iter=50000, random_state=SEED))
    svr = TransformedTargetRegressor(
        regressor=make_pipeline(imp(), StandardScaler(), pca(0.95), SVR(kernel='rbf')), transformer=StandardScaler())
    M['PCA_SVR'] = GridSearchCV(svr, {'regressor__svr__C': [0.3, 1, 3], 'regressor__svr__epsilon': [0.05, 0.2]},
                                cv=ts, scoring='neg_mean_absolute_error')
    M['PCA_KNN'] = make_pipeline(imp(), StandardScaler(), pca(0.90),
                                 KNeighborsRegressor(n_neighbors=20, weights='distance'))
    M['PCA_MLP'] = TransformedTargetRegressor(
        regressor=make_pipeline(imp(), StandardScaler(), pca(0.95),
                                MLPRegressor(hidden_layer_sizes=(32, 16), alpha=0.1, early_stopping=True,
                                             validation_fraction=0.15, n_iter_no_change=30, max_iter=3000,
                                             random_state=SEED)),
        transformer=StandardScaler())
    M['ExtraTrees'] = make_pipeline(imp(), ExtraTreesRegressor(n_estimators=300, min_samples_leaf=8, max_features=0.33,
                                                                n_jobs=-1, random_state=SEED))
    M['GBR'] = make_pipeline(imp(), GradientBoostingRegressor(n_estimators=250, learning_rate=0.03, max_depth=3,
                                                               subsample=0.7, min_samples_leaf=8, random_state=SEED))
    return M


def modelos_clf():
    imp = lambda: SimpleImputer(strategy='median')
    pca = lambda v: PCA(n_components=v, svd_solver='full', random_state=SEED)
    M = {}
    M['Mayoritaria'] = DummyClassifier(strategy='most_frequent')
    M['Logit'] = make_pipeline(imp(), StandardScaler(), LogisticRegression(C=0.05, max_iter=5000,
                                                                            class_weight='balanced'))
    M['LDA'] = make_pipeline(imp(), StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    M['PCA_LDA'] = make_pipeline(imp(), StandardScaler(), pca(0.90), LinearDiscriminantAnalysis())
    M['PCA_QDA'] = make_pipeline(imp(), StandardScaler(), pca(0.80), QuadraticDiscriminantAnalysis(reg_param=0.5))
    M['PCA_Logit'] = make_pipeline(imp(), StandardScaler(), pca(0.90), LogisticRegression(C=0.5, max_iter=5000,
                                                                                          class_weight='balanced'))
    M['RF_clf'] = make_pipeline(imp(), RandomForestClassifier(n_estimators=300, min_samples_leaf=8, max_features=0.33,
                                                              class_weight='balanced_subsample', n_jobs=-1,
                                                              random_state=SEED))
    M['HGB_clf'] = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.04, max_depth=3,
                                                  min_samples_leaf=15, l2_regularization=1.0,
                                                  class_weight='balanced', random_state=SEED)
    if QUICK:
        for k in ('PCA_QDA', 'PCA_Logit'):
            M.pop(k)
    return M

# ----------------------------------------------------------------------------- METRICAS
def dm_test(y, e, h):
    """Diebold-Mariano (con correccion de Harvey) contra el random walk. p pequeno => modelo != RW."""
    d = y ** 2 - e ** 2
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return np.nan, np.nan
    s = np.var(d)
    for k in range(1, h):
        s += 2 * np.cov(d[k:], d[:-k], ddof=0)[0, 1]
    if s <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(s / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1)))


def metricas_reg(kind, y, p, base, h):
    y, p, base = np.asarray(y, float), np.asarray(p, float), np.asarray(base, float)
    e = y - p
    pt, ph = (base * np.exp(y), base * np.exp(p)) if kind == 'logret' else (base + y, base + p)
    sse, sse0 = np.sum(e ** 2), np.sum(y ** 2)
    nz = y != 0
    pos = np.sign(p)
    r_no = (pos * y)[::h]
    ic = stats.spearmanr(p, y)[0] if np.std(p) > 0 else np.nan
    dm_s, dm_p = dm_test(y, e, h)
    return {
        'MAE_cambio': np.mean(np.abs(e)), 'RMSE_cambio': np.sqrt(np.mean(e ** 2)),
        'MAE_USD': np.mean(np.abs(pt - ph)), 'RMSE_USD': np.sqrt(np.mean((pt - ph) ** 2)),
        'Sesgo_USD': np.mean(ph - pt),
        'MAPE_pct': np.mean(np.abs(pt - ph) / np.abs(pt)) * 100 if kind == 'logret' else np.nan,
        'R2_nivel': r2_score(pt, ph),
        'R2_OOS': 1 - sse / sse0 if sse0 > 0 else np.nan,
        'Theil_U': np.sqrt(sse / sse0) if sse0 > 0 else np.nan,
        'MAE_rel_RW': np.mean(np.abs(e)) / np.mean(np.abs(y)),
        'Dir_Acc_pct': np.mean(np.sign(p[nz]) == np.sign(y[nz])) * 100 if nz.any() else np.nan,
        'IC_spearman': ic,
        'Estrat_media': r_no.mean(),
        'Estrat_Sharpe': r_no.mean() / r_no.std() * np.sqrt(12 / h) if r_no.std() > 0 else np.nan,
        'DM_stat': dm_s, 'DM_p': dm_p,
    }


def proba_completa(modelo, X):
    P = modelo.predict_proba(X)
    full = np.zeros((len(X), 3))
    cl = getattr(modelo, 'classes_', None)
    if cl is None:
        cl = modelo[-1].classes_
    for j, c in enumerate(cl):
        full[:, int(c)] = P[:, j]
    return full


def metricas_clf(y, proba):
    pred = proba.argmax(axis=1)
    onehot = np.eye(3)[y]
    try:
        auc = roc_auc_score(y, proba, multi_class='ovr', average='macro', labels=[0, 1, 2])
    except Exception:
        auc = np.nan
    return {
        'Accuracy': np.mean(pred == y), 'Balanced_Acc': balanced_accuracy_score(y, pred),
        'F1_macro': f1_score(y, pred, average='macro', zero_division=0),
        'MCC': matthews_corrcoef(y, pred),
        'LogLoss': log_loss(y, np.clip(proba, 1e-6, 1), labels=[0, 1, 2]),
        'Brier': np.mean(np.sum((proba - onehot) ** 2, axis=1)), 'AUC_OvR': auc,
    }, pred


# ----------------------------------------------------------------------------- GRANGER
def granger_enso(d):
    banner('CAUSALIDAD DE GRANGER (ENSO -> precios), sobre cambios mensuales, bloque de entrenamiento')
    if ENSO not in d.columns:
        print('(sin columna ENSO)')
        return pd.DataFrame()
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except Exception:
        print('(statsmodels no disponible)')
        return pd.DataFrame()
    rows = []
    x = d[ENSO].diff()
    for s in ['soya_grano', 'soya_aceite', 'soya_harina']:
        y = np.log(d[s]).diff()
        t = pd.concat([y.rename('y'), x.rename('x')], axis=1).dropna()
        t = t.iloc[:int(len(t) * FRAC_TRAIN)]
        for rel, cols in [(f'ENSO -> {s}', ['y', 'x']), (f'{s} -> ENSO', ['x', 'y'])]:
            try:
                try:
                    r = grangercausalitytests(t[cols], maxlag=6, verbose=False)
                except TypeError:
                    r = grangercausalitytests(t[cols], maxlag=6)
                pv = {k: r[k][0]['ssr_ftest'][1] for k in r}
                k = min(pv, key=pv.get)
                rows.append(dict(relacion=rel, mejor_rezago=k, p_min=pv[k], p_bonferroni=min(1.0, pv[k] * 6)))
            except Exception:
                rows.append(dict(relacion=rel, mejor_rezago=np.nan, p_min=np.nan, p_bonferroni=np.nan))
    g = pd.DataFrame(rows)
    print(tabla(g))
    print('p_bonferroni < 0.05 sugiere que el pasado de la primera serie ayuda a predecir la segunda.')
    return g

# ----------------------------------------------------------------------------- NUCLEO REGRESION
def evaluar_regresion(tname, h, d, X, cols, lvl, kind, M, acum):
    Xv = X.values.astype(float)
    N = len(Xv)
    chg = cambio_futuro(lvl, kind, h)
    valid = np.where(~np.isnan(Xv).any(axis=1) & ~np.isnan(chg))[0]
    tr, va, te, tv = particiones(valid, h)
    if len(tr) < 80 or len(va) < 20 or len(te) < 20:
        print(f'  [!] pocos datos para {tname} h={h} (train={len(tr)}, val={len(va)}, test={len(te)}); se omite')
        return None
    base = lvl.values.astype(float)
    banner(f'REGRESION | objetivo={tname} | horizonte={h} m | cambio={"log-retorno" if kind == "logret" else "diferencia (USD/t)"}'
           f' | train={len(tr)} val={len(va)} test={len(te)} (purga {h} m)', '-')
    print(f'Fechas: train {d.index[tr[0]].date()}->{d.index[tr[-1]].date()} | val {d.index[va[0]].date()}->'
          f'{d.index[va[-1]].date()} | test {d.index[te[0]].date()}->{d.index[te[-1]].date()}')
    print(f'Desv. estandar del cambio: train={chg[tr].std():.4f} val={chg[va].std():.4f} test={chg[te].std():.4f}')

    # ---------------- fase VALIDACION (ajuste solo con train)
    fits, pv = {}, {}
    for nm, m in M.items():
        try:
            mm = clone(m).fit(Xv[tr], chg[tr])
            fits[nm], pv[nm] = mm, mm.predict(Xv[va])
        except Exception as ex:
            print(f'  [!] {nm} fallo en validacion: {ex}')
    pv['Naive_RW'] = np.zeros(len(va))
    mae = lambda k, idx, P: float(np.mean(np.abs(chg[idx] - P[k])))
    rank = sorted([k for k in pv if k not in ('Naive_RW', 'Media_hist') and pv[k] is not None],
                  key=lambda k: mae(k, va, pv))
    top = rank[:3]
    pv['Ens_top3'] = np.mean([pv[k] for k in top], axis=0)
    cand = rank + ['Ens_top3']
    champ = min(cand, key=lambda k: mae(k, va, pv))

    # ---------------- fase TEST (reajuste con train+val)
    pt = {'Naive_RW': np.zeros(len(te))}
    for nm in fits:
        try:
            pt[nm] = clone(M[nm]).fit(Xv[tv], chg[tv]).predict(Xv[te])
        except Exception as ex:
            print(f'  [!] {nm} fallo en test: {ex}')
    top_ok = [k for k in top if k in pt]
    pt['Ens_top3'] = np.mean([pt[k] for k in top_ok], axis=0)

    filas = []
    for fase, idx, P in (('val', va, pv), ('test', te, pt)):
        for k, p in P.items():
            if p is None:
                continue
            r = metricas_reg(kind, chg[idx], p, base[idx], h)
            filas.append(dict(objetivo=tname, h=h, modelo=k, fase=fase, **r))
    R = pd.DataFrame(filas)
    cols_show = ['modelo', 'MAE_cambio', 'RMSE_cambio', 'MAE_USD', 'MAPE_pct', 'R2_nivel', 'R2_OOS', 'Theil_U',
                 'Dir_Acc_pct', 'IC_spearman', 'Estrat_Sharpe', 'DM_p']
    print('\n--- VALIDACION (ajuste con train) ---')
    print(tabla(R[R.fase == 'val'].sort_values('MAE_cambio'), cols_show))
    print(f'\nCampeon por MAE en validacion: {champ}   (ensamble top-3 = {top})')
    print('\n--- TEST (futuro puro; modelos reajustados con train+val) ---')
    print(tabla(R[R.fase == 'test'].sort_values('MAE_cambio'), cols_show))
    rt = R[(R.fase == 'test') & (R.modelo == champ)].iloc[0]
    rv = R[(R.fase == 'val') & (R.modelo == champ)].iloc[0]
    print(f'\n>> Campeon {champ}: TEST MAE_USD={rt.MAE_USD:,.2f} | R2_OOS={rt.R2_OOS:+.3f} | Theil_U={rt.Theil_U:.3f} | '
          f'DirAcc={rt.Dir_Acc_pct:.1f}% | DM_p={rt.DM_p:.3f}  ->  '
          f'{"SUPERA" if rt.Theil_U < 1 else "NO supera"} al random walk')

    # ---------------- intervalos conformales (calibrados con el modelo REAJUSTADO sobre val)
    q, cobertura, ancho = np.nan, np.nan, np.nan
    try:
        if champ in M:
            m_refit = clone(M[champ]).fit(Xv[tv], chg[tv])
            res_cal = chg[va] - m_refit.predict(Xv[va])
        else:  # Ensamble: no hay objeto reentrenable; usa sus residuos de validacion
            res_cal = chg[va] - pv[champ]
        q = float(np.quantile(np.abs(res_cal), min(1.0, (1 - ALPHA) * (1 + 1 / len(va)))))
        lo, hi = pt[champ] - q, pt[champ] + q
        cobertura = float(np.mean((chg[te] >= lo) & (chg[te] <= hi)) * 100)
        ancho = float(np.mean(base[te] * (np.exp(hi) - np.exp(lo)))) if kind == 'logret' else float(np.mean(hi - lo))
        print(f'Intervalo conformal {int((1 - ALPHA) * 100)}% (calibrado con refit): q={q:.4f} | '
              f'cobertura real en test={cobertura:.1f}% | ancho medio={ancho:,.2f} USD')
    except Exception as ex:
        print(f'  [!] conformal fallo: {ex}')

    # ---------------- importancia por permutacion (mejor modelo individual, sobre validacion)
    indiv = [k for k in rank if k in fits]
    imp_df = pd.DataFrame()
    if indiv:
        mejor = indiv[0]
        try:
            r = permutation_importance(fits[mejor], Xv[va], chg[va], scoring='neg_mean_absolute_error',
                                       n_repeats=8, random_state=SEED, n_jobs=1)
            imp = pd.Series(r.importances_mean, index=cols).sort_values(ascending=False)
            grp = imp.clip(lower=0).groupby([c.split('__')[0] for c in imp.index]).sum()
            grp = (grp / grp.sum() * 100).sort_values(ascending=False) if grp.sum() > 0 else grp
            print(f'\nImportancia por permutacion ({mejor}, validacion) - top 10 variables (delta MAE):')
            print('  ' + ' | '.join(f'{k}={v:.5f}' for k, v in imp.head(10).items()))
            print('Importancia por grupo (% del total): ' + ' | '.join(f'{k}={v:.1f}' for k, v in grp.items()))
            imp_df = pd.DataFrame({'objetivo': tname, 'h': h, 'modelo': mejor, 'variable': imp.index[:15],
                                   'delta_MAE': imp.values[:15]})
            for g, v in grp.items():
                acum['grupos'].append(dict(objetivo=tname, h=h, modelo=mejor, grupo=g, pct=float(v)))
            acum['imp'].append(imp_df)
        except Exception as ex:
            print(f'  [!] importancia fallo: {ex}')

    # ---------------- ablacion ENSO
    idx_noenso = [i for i, c in enumerate(cols) if not c.startswith('enso__')]
    if len(idx_noenso) < len(cols):
        print('\nValor incremental de ENSO (ablacion; delta% > 0 => ENSO reduce el error):')
        for nm in ('HGB', 'PCA_Ridge'):
            if nm not in M or nm not in fits:
                continue
            try:
                a_val = np.mean(np.abs(chg[va] - clone(M[nm]).fit(Xv[tr][:, idx_noenso], chg[tr]).predict(Xv[va][:, idx_noenso])))
                a_te = np.mean(np.abs(chg[te] - clone(M[nm]).fit(Xv[tv][:, idx_noenso], chg[tv]).predict(Xv[te][:, idx_noenso])))
                c_val, c_te = mae(nm, va, pv), mae(nm, te, pt)
                fila = dict(objetivo=tname, h=h, modelo=nm, MAE_val_con=c_val, MAE_val_sin=a_val,
                            MAE_test_con=c_te, MAE_test_sin=a_te,
                            delta_val_pct=(a_val - c_val) / a_val * 100, delta_test_pct=(a_te - c_te) / a_te * 100)
                acum['enso'].append(fila)
                print(f'  {nm}: val {a_val:.4f}->{c_val:.4f} ({fila["delta_val_pct"]:+.1f}%) | '
                      f'test {a_te:.4f}->{c_te:.4f} ({fila["delta_test_pct"]:+.1f}%)')
            except Exception as ex:
                print(f'  [!] ablacion {nm} fallo: {ex}')

    # ---------------- pronostico prospectivo (todas las filas etiquetadas)
    try:
        ult = N - 1
        if np.isnan(Xv[ult]).any():
            raise ValueError('ultima fila de variables incompleta')
        miembros = top_ok if champ == 'Ens_top3' else [champ]
        preds = []
        for k in miembros:
            preds.append(float(clone(M[k]).fit(Xv[valid], chg[valid]).predict(Xv[ult:ult + 1])[0]))
        pc = float(np.mean(preds))
        p0 = float(base[ult])
        if kind == 'logret':
            p_hat, p_lo, p_hi = p0 * np.exp(pc), p0 * np.exp(pc - q), p0 * np.exp(pc + q)
        else:
            p_hat, p_lo, p_hi = p0 + pc, p0 + pc - q, p0 + pc + q
        senal = 'ALZA' if pc - q > 0 else ('BAJA' if pc + q < 0 else 'SIN SENAL (IC incluye 0)')
        fila = dict(objetivo=tname, h=h, modelo=champ, fecha_origen=d.index[ult].date(),
                    fecha_objetivo=(d.index[ult] + pd.DateOffset(months=h)).date(), nivel_actual=p0,
                    pronostico=p_hat, ic80_inf=p_lo, ic80_sup=p_hi, cambio_esperado_pct=(p_hat / p0 - 1) * 100,
                    senal=senal, supera_RW_test=bool(rt.Theil_U < 1))
        acum['pron'].append(fila)
        print(f'\nPronostico prospectivo ({champ}): {fila["fecha_origen"]} -> {fila["fecha_objetivo"]}: '
              f'{p0:,.1f} -> {p_hat:,.1f} (IC80 {p_lo:,.1f} a {p_hi:,.1f}) | senal: {senal}')
    except Exception as ex:
        print(f'  [!] pronostico prospectivo no disponible: {ex}')

    rv_ = rv
    acum['reg'].append(R)
    acum['camp'].append(dict(objetivo=tname, h=h, campeon=champ, MAE_val=rv_.MAE_cambio, MAE_test=rt.MAE_cambio,
                             MAE_USD_test=rt.MAE_USD, MAPE_pct_test=rt.MAPE_pct, R2_nivel_test=rt.R2_nivel,
                             R2_OOS_test=rt.R2_OOS, Theil_U_test=rt.Theil_U, Dir_Acc_pct_test=rt.Dir_Acc_pct,
                             IC_test=rt.IC_spearman, Sharpe_test=rt.Estrat_Sharpe, DM_p_test=rt.DM_p,
                             cobertura80_pct=cobertura, ancho_ic_USD=ancho))
    return dict(valid=valid, tr=tr, va=va, te=te, tv=tv, chg=chg)


# ----------------------------------------------------------------------------- NUCLEO CLASIFICACION
def evaluar_clasificacion(tname, h, X, cols, ctx, MC, acum):
    Xv = X.values.astype(float)
    tr, va, te, tv, chg = ctx['tr'], ctx['va'], ctx['te'], ctx['tv'], ctx['chg']
    thr = 0.5 * np.std(chg[tr])
    cls = np.where(chg < -thr, 0, np.where(chg > thr, 2, 1))
    banner(f'CLASIFICACION DE DIRECCION | objetivo={tname} | h={h} m | umbral=+-{thr:.4f} (0.5 desv. de train)', '-')
    dist = lambda idx: ' '.join(f'{CLASES[c]}={np.mean(cls[idx] == c) * 100:.0f}%' for c in (0, 1, 2))
    print(f'Distribucion de clases: train[{dist(tr)}] val[{dist(va)}] test[{dist(te)}]')
    fits, pv = {}, {}
    for nm, m in MC.items():
        try:
            mm = clone(m).fit(Xv[tr], cls[tr])
            fits[nm] = mm
            pv[nm] = proba_completa(mm, Xv[va])
        except Exception as ex:
            print(f'  [!] {nm} fallo en validacion: {ex}')
    resv = {k: metricas_clf(cls[va], p)[0] for k, p in pv.items()}
    campeon = max([k for k in resv if k != 'Mayoritaria'], key=lambda k: (resv[k]['Balanced_Acc'], resv[k]['MCC']))
    filas, preds_te = [], {}
    for nm in fits:
        try:
            mm = clone(MC[nm]).fit(Xv[tv], cls[tv])
            pr = proba_completa(mm, Xv[te])
            m_te, pred = metricas_clf(cls[te], pr)
            preds_te[nm] = pred
            filas.append(dict(objetivo=tname, h=h, modelo=nm, fase='test', **m_te))
            filas.append(dict(objetivo=tname, h=h, modelo=nm, fase='val', **resv[nm]))
        except Exception as ex:
            print(f'  [!] {nm} fallo en test: {ex}')
    Rc = pd.DataFrame(filas)
    cs = ['modelo', 'Accuracy', 'Balanced_Acc', 'F1_macro', 'MCC', 'LogLoss', 'Brier', 'AUC_OvR']
    print('\n--- VALIDACION ---')
    print(tabla(Rc[Rc.fase == 'val'].sort_values('Balanced_Acc', ascending=False), cs))
    print(f'\nCampeon por Balanced_Acc en validacion: {campeon}')
    print('\n--- TEST (modelos reajustados con train+val) ---')
    print(tabla(Rc[Rc.fase == 'test'].sort_values('Balanced_Acc', ascending=False), cs))
    cm = confusion_matrix(cls[te], preds_te[campeon], labels=[0, 1, 2])
    print(f'\nMatriz de confusion TEST ({campeon}) filas=real, columnas=predicho [BAJA, LATERAL, ALZA]:')
    print(cm)
    k_ok = int(np.sum(preds_te[campeon] == cls[te]))
    maj = max(np.mean(cls[te] == c) for c in (0, 1, 2))
    try:
        pbin = float(stats.binomtest(k_ok, len(te), maj, alternative='greater').pvalue)
    except Exception:
        pbin = np.nan
    print(f'Accuracy test={k_ok / len(te) * 100:.1f}% vs clase mayoritaria={maj * 100:.1f}% | p(binomial, acc>mayoritaria)={pbin:.3f}')
    try:
        Zs = StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(Xv[tr]))
        lda = LinearDiscriminantAnalysis(solver='svd').fit(Zs, cls[tr])
        evr = lda.explained_variance_ratio_
        sc = pd.Series(lda.scalings_[:, 0], index=cols)
        top = sc.reindex(sc.abs().sort_values(ascending=False).index)[:6]
        print('Ejes LDA, varianza discriminante (%):', np.round(evr * 100, 1).tolist(),
              '| top cargas LD1: ' + ' | '.join(f'{k}={v:+.2f}' for k, v in top.items()))
    except Exception:
        pass
    # calibracion del campeon (sigmoid con CV interna sobre train) con Brier/LogLoss antes/despues
    try:
        from sklearn.calibration import CalibratedClassifierCV
        base = clone(MC[campeon]).fit(Xv[tv], cls[tv])
        cal = CalibratedClassifierCV(estimator=clone(MC[campeon]), method='sigmoid', cv=3)
        cal.fit(Xv[tr], cls[tr])
        for tag, Xx in (('sin-calibrar', base), ('calibrado', cal)):
            pr = proba_completa(Xx, Xv[te])
            onehot = np.eye(3)[cls[te]]
            print(f'  {tag}: Brier={np.mean(np.sum((pr - onehot) ** 2, axis=1)):.4f} '
                  f'LogLoss={log_loss(cls[te], np.clip(pr, 1e-6, 1), labels=[0, 1, 2]):.4f}')
    except Exception as ex:
        print(f'  [!] calibracion fallo: {ex}')
    rt = Rc[(Rc.fase == 'test') & (Rc.modelo == campeon)].iloc[0]
    acum['clf'].append(Rc)
    acum['campclf'].append(dict(objetivo=tname, h=h, campeon=campeon, Accuracy=rt.Accuracy, Balanced_Acc=rt.Balanced_Acc,
                                F1_macro=rt.F1_macro, MCC=rt.MCC, AUC_OvR=rt.AUC_OvR, LogLoss=rt.LogLoss,
                                clase_mayoritaria_pct=maj * 100, p_binomial=pbin))

# ----------------------------------------------------------------------------- MAIN
def main():
    t0 = time.time()
    print('Librerias:', versiones())
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        sys.stdout = Tee(os.path.join(OUT_DIR, 'log_completo.txt'))
    except Exception:
        pass
    banner('PROYECTO 1 v2 | PRONOSTICO DE PRECIOS DE SOYA Y MARGEN DE CRUSH')
    print(f'Config: horizontes={HORIZONTES} objetivos={TARGETS} quick={QUICK} semilla={SEED}')
    be = Backend()
    d = preparar(be.cargar_precios())
    X = construir_features(d)
    cols = list(X.columns)
    grupos = pd.Series([c.split('__')[0] for c in cols]).value_counts()
    print(f'\nVariables construidas: {len(cols)} | por grupo: ' + ' | '.join(f'{k}={v}' for k, v in grupos.items()))
    print(f'Filas disponibles: {len(X)} (las primeras ~24 se pierden por ventanas moviles)')
    targets = [t for t in TARGETS if t == 'crush_spread' or t in d.columns]
    if not targets:
        raise SystemExit('Ningun objetivo valido en TARGETS')

    chg3 = cambio_futuro(d['soya_grano'], 'logret', 3)
    ok0 = np.where(~np.isnan(X.values).any(axis=1) & ~np.isnan(chg3))[0]
    n_tr = int(len(ok0) * FRAC_TRAIN)
    pca_info = analisis_pca(X.values[ok0[:n_tr]], cols, chg3[ok0[:n_tr]])
    granger = granger_enso(d)

    M, MC = modelos_reg(), modelos_clf()
    try:
        import statsmodels  # noqa: F401 (solo para Granger; no hay modelos ARIMA)
    except Exception:
        print('\n[!] statsmodels no disponible: se omite Granger')
    acum = dict(reg=[], clf=[], camp=[], campclf=[], imp=[], grupos=[], enso=[], pron=[])
    for tname in targets:
        lvl, kind = serie_objetivo(d, tname)
        for h in HORIZONTES:
            ctx = evaluar_regresion(tname, h, d, X, cols, lvl, kind, M, acum)
            if ctx is not None:
                evaluar_clasificacion(tname, h, X, cols, ctx, MC, acum)

    cc = lambda k: pd.concat(acum[k], ignore_index=True) if acum[k] else pd.DataFrame()
    df_reg, df_clf, df_imp = cc('reg'), cc('clf'), cc('imp')
    df_camp, df_campclf = pd.DataFrame(acum['camp']), pd.DataFrame(acum['campclf'])
    df_grp, df_enso, df_pron = pd.DataFrame(acum['grupos']), pd.DataFrame(acum['enso']), pd.DataFrame(acum['pron'])

    banner('RESUMEN_PARA_COMPARTIR (copia desde aqui)')
    L = []
    L.append(f'Datos: {len(d)} meses {d.index.min().date()}->{d.index.max().date()} | variables={len(cols)} | '
             f'PCA comps para 80/90/95/99% = {pca_info["ncomp"]} | PC1-3 var(%) = {np.round(pca_info["ev"][:3] * 100, 1).tolist()} | '
             f'cond(X)={pca_info["cond"]:,.0f}')
    L.append('\n[A] CAMPEONES REGRESION (test)')
    L.append(tabla(df_camp, ['objetivo', 'h', 'campeon', 'MAE_val', 'MAE_test', 'MAE_USD_test', 'MAPE_pct_test',
                             'R2_nivel_test', 'R2_OOS_test', 'Theil_U_test', 'Dir_Acc_pct_test', 'IC_test',
                             'Sharpe_test', 'DM_p_test', 'cobertura80_pct', 'ancho_ic_USD']))
    if len(df_reg):
        t = df_reg[df_reg.fase == 'test'].copy()
        L.append('\n[B] MAE_cambio en TEST por modelo (filas=modelo, columnas=objetivo_h)')
        piv = t.pivot_table(index='modelo', columns=['objetivo', 'h'], values='MAE_cambio')
        L.append(piv.round(4).to_string())
        L.append('\n[C] R2_OOS en TEST por modelo (>0 supera al random walk)')
        L.append(t.pivot_table(index='modelo', columns=['objetivo', 'h'], values='R2_OOS').round(3).to_string())
        L.append('\n[D] Acierto direccional % en TEST por modelo')
        L.append(t.pivot_table(index='modelo', columns=['objetivo', 'h'], values='Dir_Acc_pct').round(1).to_string())
    L.append('\n[E] CLASIFICACION campeones (test)')
    L.append(tabla(df_campclf))
    if len(df_clf):
        tc = df_clf[df_clf.fase == 'test']
        L.append('\n[F] Balanced_Acc en TEST por modelo')
        L.append(tc.pivot_table(index='modelo', columns=['objetivo', 'h'], values='Balanced_Acc').round(3).to_string())
    L.append('\n[G] VALOR INCREMENTAL DE ENSO (ablacion)')
    L.append(tabla(df_enso))
    L.append('\n[H] IMPORTANCIA POR GRUPO (% del total, mejor modelo individual)')
    if len(df_grp):
        L.append(df_grp.pivot_table(index='grupo', columns=['objetivo', 'h'], values='pct').round(1).to_string())
    L.append('\n[I] GRANGER ENSO <-> PRECIOS')
    L.append(tabla(granger))
    L.append('\n[J] PRONOSTICO PROSPECTIVO')
    L.append(tabla(df_pron, nd=2))
    # DM con Holm por familia (objetivo, h): que victorias sobreviven correccion multiple
    try:
        th = df_reg[df_reg.fase == 'test'][['objetivo', 'h', 'modelo', 'DM_p']].copy()
        th['DM_rechaza_raw'] = th['DM_p'] < 0.05
        outs = []
        for (o, h), g in th.groupby(['objetivo', 'h']):
            rej = holm(g['DM_p'].values)
            g = g.copy()
            g['DM_Holm'] = rej
            outs.append(g)
        th = pd.concat(outs, ignore_index=True)
        L.append('\n[K] DIEBOLD-MARIANO CON HOLM por familia (objetivo,h): victorias que sobreviven')
        L.append(tabla(th[th['DM_Holm']][['objetivo', 'h', 'modelo', 'DM_p']]))
        df_dmholm = th
    except Exception as ex:
        print(f'  [!] Holm fallo: {ex}')
        df_dmholm = pd.DataFrame()
    L.append(f'\nTiempo total: {(time.time() - t0) / 60:.1f} min')
    resumen = '\n'.join(L)
    print(resumen)
    print('\n' + '=' * 100 + '\nFIN RESUMEN_PARA_COMPARTIR\n' + '=' * 100)

    try:
        for nom, df in [('metricas_regresion', df_reg), ('metricas_clasificacion', df_clf),
                        ('campeones_regresion', df_camp), ('campeones_clasificacion', df_campclf),
                        ('importancia_variables', df_imp), ('importancia_grupos', df_grp),
                        ('ablacion_enso', df_enso), ('granger', granger), ('pronostico', df_pron),
                        ('dm_holm', df_dmholm)]:
            if len(df):
                df.to_csv(os.path.join(OUT_DIR, f'{nom}.csv'), index=False, encoding='utf-8')
        with open(os.path.join(OUT_DIR, 'RESUMEN_PARA_COMPARTIR.txt'), 'w', encoding='utf-8') as f:
            f.write(resumen)
        print(f'\nArchivos guardados en la carpeta: {os.path.abspath(OUT_DIR)}')
    except Exception as ex:
        print(f'[!] no se pudieron guardar archivos locales: {ex}')
    if ESCRIBIR:
        be.escribir(df_reg, 'SOYA_PRECIOS_METRICAS_REG')
        be.escribir(df_clf, 'SOYA_PRECIOS_METRICAS_CLF')
        be.escribir(df_pron, 'SOYA_PRECIOS_PRONOSTICO')
        be.escribir(df_imp, 'SOYA_PRECIOS_IMPORTANCIA')
        be.escribir(df_dmholm, 'SOYA_PRECIOS_DM_HOLM')
    if hasattr(sys.stdout, 'f'):
        sys.stdout = sys.stdout.o


if __name__ == '__main__':
    main()
