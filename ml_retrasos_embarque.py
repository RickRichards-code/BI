#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROYECTO 4 | Prediccion de retrasos en envios y embarques (demurrage)
=====================================================================
Fuente : ARQUI1.CAPA_BRONCE (EMBARQUE + ENVIO + TRANSPORTE + PUERTO + VENTA +
         CONTRATO_VENTA + CLIENTE). Solo ML (Naive/Media = referencias).
Objetivos : (a) severidad 3 clases (0 / 1-5 / 6+ dias), (b) dias de retraso (regresion).
Excluye FECHA_REAL nula (censura: 10.5%). Split TEMPORAL 70/15/15 por FECHA_PROGRAMADA.
Seleccion en validacion (F1-macro / MAE), test unico. Panel de riesgo ruta x transportista
con costo de sobrestadia (DEMURRAGE_USD/dia, supuesto). Pronostico prospectivo + resumen.
Uso Snowflake (sin credenciales) o local: export SNOWFLAKE_PASSWORD='clave'.
QUICK=1 prueba rapida. ESCRIBIR=1 guarda predicciones. OUT_DIR=resultados_retrasos
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
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingRegressor,
                              HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             log_loss, matthews_corrcoef, mean_absolute_error, mean_squared_error,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedShuffleSplit, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
SEED = 42
np.random.seed(SEED)
QUICK = os.getenv('QUICK', '0') == '1'
OUT_DIR = os.getenv('OUT_DIR', 'resultados_retrasos')
ESCRIBIR = os.getenv('ESCRIBIR', '0') == '1'
DEMURRAGE = float(os.getenv('DEMURRAGE_USD', '5000'))
FRAC_TRAIN, FRAC_VAL = 0.70, 0.15
ALPHA = 0.20
SEV = {0: 'A_TIEMPO', 1: 'LEVE_1_5', 2: 'GRAVE_6_MAS'}


def versiones():
    mods = {}
    for nom in ('numpy', 'pandas', 'scipy', 'sklearn'):
        try:
            mods[nom] = __import__('importlib').import_module(nom).__version__
        except Exception:
            mods[nom] = 'no-instalado'
    return mods


class Tee:
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

    def cargar(self):
        q = """SELECT E.ID_EMBARQUE, E.ID_VENTA, E.ID_PUERTO, E.FECHA_PROGRAMADA, E.FECHA_REAL,
            E.CANTIDAD_TN, E.NAVE_BARCAZA, E.ESTADO AS ESTADO_EMBARQUE,
            P.NOMBRE AS PUERTO, P.TIPO_PUERTO, P.RIO_MAR,
            V.ID_CONTRATO_VENTA, V.ID_PRODUCTO, V.CANTIDAD_TN AS VENTA_TN,
            CV.INCOTERM, CV.PUERTO_EMBARQUE, CV.FECHA AS FECHA_CONTRATO,
            C.PAIS AS PAIS_CLIENTE, C.SEGMENTO AS SEG_CLIENTE,
            V2.N_ENVIOS, V2.DISTANCIA_TOT, V2.FLETE_TOT, V2.TIPO_DOM, V2.DIAS_TRANSITO_MED
            FROM ARQUI1.CAPA_BRONCE.EMBARQUE E
            LEFT JOIN ARQUI1.CAPA_BRONCE.PUERTO P ON E.ID_PUERTO = P.ID_PUERTO
            LEFT JOIN ARQUI1.CAPA_BRONCE.VENTA V ON E.ID_VENTA = V.ID_VENTA
            LEFT JOIN ARQUI1.CAPA_BRONCE.CONTRATO_VENTA CV ON V.ID_CONTRATO_VENTA = CV.ID_CONTRATO_VENTA
            LEFT JOIN ARQUI1.CAPA_BRONCE.CLIENTE C ON CV.ID_CLIENTE = C.ID_CLIENTE
            LEFT JOIN (SELECT ID_EMBARQUE, COUNT(*) AS N_ENVIOS, SUM(DISTANCIA_KM) AS DISTANCIA_TOT,
                SUM(COSTO_FLETE_USD) AS FLETE_TOT,
                MAX_BY(T.TIPO, V.DISTANCIA_KM) AS TIPO_DOM,
                AVG(DATEDIFF('hour', V.FECHA_SALIDA, V.FECHA_LLEGADA)) AS DIAS_TRANSITO_MED
                FROM ARQUI1.CAPA_BRONCE.ENVIO V
                LEFT JOIN ARQUI1.CAPA_BRONCE.TRANSPORTE T ON V.ID_TRANSPORTE = T.ID_TRANSPORTE
                GROUP BY 1) V2 ON V2.ID_EMBARQUE = E.ID_EMBARQUE"""
        if self.mode == 'csv':
            return pd.read_csv(os.getenv('DATA_CSV'))
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


# ----------------------------------------------------------------------------- FEATURES
NUM = ['CANTIDAD_TN', 'VENTA_TN', 'DISTANCIA_TOT', 'FLETE_TOT', 'N_ENVIOS',
       'REZAGO_CONTRATO_D', 'CONGESTION_PUERTO_7D', 'HIST_PUNT_TIPO']
CAT = ['PUERTO', 'TIPO_PUERTO', 'RIO_MAR', 'INCOTERM', 'SEG_CLIENTE', 'PAIS_CLIENTE',
       'TIPO_DOM', 'MES', 'ID_PRODUCTO']


def preparar(df):
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    for c in ['fecha_programada', 'fecha_real', 'fecha_contrato']:
        df[c] = pd.to_datetime(df[c], errors='coerce')
    n0 = len(df)
    cens = int(df['fecha_real'].isna().sum())
    print(f'Embarques: {n0} | sin FECHA_REAL (censura, se excluyen del supervisado): {cens}')
    df = df.dropna(subset=['fecha_real']).copy()
    df['retraso_d'] = (df['fecha_real'] - df['fecha_programada']).dt.days
    print('Retraso dias: describe:')
    print(df['retraso_d'].describe().round(1).to_string())
    print('Cobertura joins: sin envio=%.1f%%' % (df['n_envios'].isna().mean() * 100))
    df = df.sort_values('fecha_programada').reset_index(drop=True)
    df['rezago_contrato_d'] = (df['fecha_programada'] - df['fecha_contrato']).dt.days
    # congestion: embarques del mismo puerto en los 7 dias previos (solo pasado, sin fuga)
    df['congestion_puerto_7d'] = 0
    for pto, g in df.groupby('id_puerto'):
        f = pd.to_datetime(df.loc[g.index, 'fecha_programada']).values.astype('datetime64[D]').astype(int)
        cnt = np.array([int(np.sum((f >= d - 7) & (f < d))) for d in f])
        df.loc[g.index, 'congestion_puerto_7d'] = cnt
    # historial de puntualidad por tipo de transporte (expanding, desplazado: sin fuga)
    df['a_tiempo'] = (df['retraso_d'] <= 0).astype(float)
    df['hist_punt_tipo'] = (df.groupby('tipo_dom')['a_tiempo'].cumsum().shift(1).fillna(0.5)
                            / df.groupby('tipo_dom').cumcount().replace(0, np.nan).shift(1).fillna(1)).clip(0, 1).fillna(0.5)
    df['mes'] = df['fecha_programada'].dt.month.astype(str)
    for c in ['cantidad_tn', 'venta_tn', 'distancia_tot', 'flete_tot', 'n_envios', 'rezago_contrato_d',
              'congestion_puerto_7d', 'hist_punt_tipo']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ['puerto', 'tipo_puerto', 'rio_mar', 'incoterm', 'seg_cliente', 'pais_cliente',
              'tipo_dom', 'mes', 'id_producto']:
        df[c] = df[c].fillna('FALTA').astype(str)
    df['y_sev'] = np.where(df['retraso_d'] <= 0, 0, np.where(df['retraso_d'] <= 5, 1, 2))
    print('Clases [a_tiempo/leve/grave]:', pd.Series(df['y_sev']).value_counts().sort_index().to_dict())
    return df


def split_temporal(df):
    n = len(df)
    a, b = int(n * FRAC_TRAIN), int(n * (FRAC_TRAIN + FRAC_VAL))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


# ----------------------------------------------------------------------------- MODELOS
def pre():
    return ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()),
         [x.lower() for x in NUM]),
        ('cat', make_pipeline(SimpleImputer(strategy='constant', fill_value='FALTA'),
                               OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
         [x.lower() for x in CAT])])


def pre_sparse():
    return ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), [x.lower() for x in NUM]),
        ('cat', make_pipeline(SimpleImputer(strategy='constant', fill_value='FALTA'),
                               OneHotEncoder(handle_unknown='ignore', sparse_output=False)), [x.lower() for x in CAT])])


def modelos_clf():
    M = {}
    M['Mayoritaria'] = DummyClassifier(strategy='most_frequent')
    M['Logit'] = make_pipeline(pre(), LogisticRegression(C=0.05, max_iter=5000, class_weight='balanced'))
    M['LDA'] = make_pipeline(pre(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    M['RF_clf'] = make_pipeline(pre(), RandomForestClassifier(n_estimators=300, min_samples_leaf=8,
                                                               class_weight='balanced_subsample', n_jobs=-1,
                                                               random_state=SEED))
    M['HGB_clf'] = make_pipeline(pre_sparse(), HistGradientBoostingClassifier(max_iter=150, learning_rate=0.04, max_depth=3,
                                                  min_samples_leaf=15, l2_regularization=1.0,
                                                  class_weight='balanced', random_state=SEED))
    if not QUICK:
        M['QDA'] = make_pipeline(pre(), QuadraticDiscriminantAnalysis(reg_param=0.5))
        M['KNN'] = make_pipeline(pre(), KNeighborsClassifier(n_neighbors=25, weights='distance'))
    return M


def modelos_reg():
    M = {}
    M['Media'] = DummyRegressor(strategy='mean')
    M['Ridge'] = make_pipeline(pre(), RidgeCV(alphas=np.logspace(-1, 5, 25)))
    M['RF'] = make_pipeline(pre(), RandomForestRegressor(n_estimators=300, min_samples_leaf=8, n_jobs=-1,
                                                          random_state=SEED))
    M['HGB'] = make_pipeline(pre_sparse(), HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_depth=4,
                                             min_samples_leaf=10, l2_regularization=1.0, random_state=SEED))
    if not QUICK:
        M['ET'] = make_pipeline(pre(), ExtraTreesRegressor(n_estimators=300, min_samples_leaf=8, n_jobs=-1,
                                                           random_state=SEED))
    return M


# ----------------------------------------------------------------------------- EVALUACION
def proba3(modelo, X):
    P = modelo.predict_proba(X)
    try:
        cl = list(modelo.classes_)
    except Exception:
        cl = list(modelo[-1].classes_)
    full = np.zeros((len(X), 3))
    for j, c in enumerate(cl):
        full[:, int(c)] = P[:, j]
    return full


def met_clf(y, proba, grave_idx=2):
    from sklearn.metrics import average_precision_score
    pred = proba.argmax(axis=1)
    onehot = np.eye(3)[y]
    try:
        auc = roc_auc_score(y, proba, multi_class='ovr', average='macro', labels=[0, 1, 2])
    except Exception:
        auc = np.nan
    try:
        ap = average_precision_score((y == grave_idx).astype(int), proba[:, grave_idx])
    except Exception:
        ap = np.nan
    return {'Accuracy': accuracy_score(y, pred), 'Balanced_Acc': balanced_accuracy_score(y, pred),
            'F1_macro': f1_score(y, pred, average='macro', zero_division=0), 'MCC': matthews_corrcoef(y, pred),
            'Recall_grave': recall_score(y, pred, labels=[grave_idx], average='macro', zero_division=0),
            'PR_AUC_grave': ap, 'LogLoss': log_loss(y, np.clip(proba, 1e-6, 1), labels=[0, 1, 2]),
            'Brier': np.mean(np.sum((proba - onehot) ** 2, axis=1)), 'AUC_OvR': auc}, pred


def met_reg(y, p):
    e = y - p
    return {'MAE': mean_absolute_error(y, p), 'RMSE': float(np.sqrt(mean_squared_error(y, p))),
            'R2': r2_score(y, p),
            'MAE_grave': float(np.mean(np.abs(e[y >= 6]))) if (y >= 6).any() else np.nan}


def evaluar_clf(tr, va, te, MC, acum):
    feats = [x.lower() for x in NUM + CAT]
    Xtr, ytr, Xva, yva, Xte, yte = tr[feats], tr['y_sev'].values, va[feats], va['y_sev'].values, te[feats], te['y_sev'].values
    banner('CLASIFICACION severidad [a_tiempo/leve/grave]', '-')
    fits, pv = {}, {}
    for nm, m in MC.items():
        try:
            mm = clone(m).fit(Xtr, ytr)
            fits[nm] = mm
            pv[nm] = proba3(mm, Xva)
        except Exception as ex:
            print(f'  [!] {nm} fallo en validacion: {ex}')
    resv = {k: met_clf(yva, p)[0] for k, p in pv.items()}
    camp = max([k for k in resv if k != 'Mayoritaria'], key=lambda k: (resv[k]['F1_macro'], resv[k]['Recall_grave']))
    print('\n--- VALIDACION ---')
    Rv = pd.DataFrame(resv).T
    print(tabla(Rv.reset_index().rename(columns={'index': 'modelo'}),
                ['modelo', 'Accuracy', 'Balanced_Acc', 'F1_macro', 'MCC', 'Recall_grave', 'PR_AUC_grave', 'AUC_OvR']))
    print(f'\nCampeon por F1-macro: {camp}')
    filas, preds = [], {}
    for nm in fits:
        try:
            mm = clone(MC[nm]).fit(pd.concat([Xtr, Xva]), np.concatenate([ytr, yva]))
            pr = proba3(mm, Xte)
            m_te, pred = met_clf(yte, pr)
            preds[nm] = pred
            filas.append(dict(modelo=nm, fase='test', **m_te))
            filas.append(dict(modelo=nm, fase='val', **resv[nm]))
        except Exception as ex:
            print(f'  [!] {nm} fallo en test: {ex}')
    Rc = pd.DataFrame(filas)
    print('\n--- TEST ---')
    print(tabla(Rc[Rc.fase == 'test'].sort_values('F1_macro', ascending=False),
                ['modelo', 'Accuracy', 'Balanced_Acc', 'F1_macro', 'MCC', 'Recall_grave', 'PR_AUC_grave', 'AUC_OvR']))
    cm = confusion_matrix(yte, preds[camp], labels=[0, 1, 2])
    print(f'\nMatriz TEST ({camp}) filas=real [a_tiempo/leve/grave]:\n{cm}')
    try:
        from sklearn.calibration import CalibratedClassifierCV
        base = clone(MC[camp]).fit(pd.concat([Xtr, Xva]), np.concatenate([ytr, yva]))
        cal = CalibratedClassifierCV(estimator=clone(MC[camp]), method='sigmoid', cv=3)
        cal.fit(Xtr, ytr)
        for tag, Xx in (('sin-calibrar', base), ('calibrado', cal)):
            pr = proba3(Xx, Xte)
            print(f'  {tag}: Brier={np.mean(np.sum((pr - np.eye(3)[yte]) ** 2, axis=1)):.4f}')
    except Exception as ex:
        print(f'  [!] calibracion fallo: {ex}')
    rt = Rc[(Rc.fase == 'test') & (Rc.modelo == camp)].iloc[0]
    acum['clf'].append(Rc)
    acum['campclf'].append(dict(campeon=camp, Accuracy=rt.Accuracy, Balanced_Acc=rt.Balanced_Acc,
                                F1_macro=rt.F1_macro, MCC=rt.MCC, Recall_grave=rt.Recall_grave,
                                PR_AUC_grave=rt.PR_AUC_grave, AUC_OvR=rt.AUC_OvR))
    # importancia del campeon (si expone importancias)
    try:
        est = fits[camp]
        imp = None
        for step in (est.steps if hasattr(est, 'steps') else []):
            if hasattr(step[1], 'feature_importances_'):
                imp = step[1].feature_importances_
                break
        if imp is not None:
            names = list(est[:-1].get_feature_names_out())
            top = pd.Series(imp, index=names).sort_values(ascending=False).head(10)
            print('Importancia (campeon val):\n' + top.round(4).to_string())
            acum['imp'].append(pd.DataFrame({'variable': top.index, 'importancia': top.values}))
    except Exception as ex:
        print(f'  [!] importancia fallo: {ex}')
    return preds[camp]


def evaluar_reg(tr, va, te, MR, acum):
    feats = [x.lower() for x in NUM + CAT]
    Xtr, ytr, Xva, yva, Xte, yte = tr[feats], tr['retraso_d'].values, va[feats], va['retraso_d'].values, te[feats], te['retraso_d'].values
    banner('REGRESION dias de retraso', '-')
    fits, pv = {}, {}
    for nm, m in MR.items():
        try:
            mm = clone(m).fit(Xtr, ytr)
            fits[nm] = mm
            pv[nm] = mm.predict(Xva)
        except Exception as ex:
            print(f'  [!] {nm} fallo en validacion: {ex}')
    resv = {k: met_reg(yva, p) for k, p in pv.items()}
    camp = min([k for k in resv if k != 'Media'], key=lambda k: resv[k]['MAE'])
    print('\n--- VALIDACION ---')
    print(tabla(pd.DataFrame(resv).T.reset_index().rename(columns={'index': 'modelo'}),
                ['modelo', 'MAE', 'RMSE', 'R2', 'MAE_grave']))
    print(f'\nCampeon por MAE en validacion: {camp}')
    filas = {}
    for nm in fits:
        try:
            mm = clone(MR[nm]).fit(pd.concat([Xtr, Xva]), np.concatenate([ytr, yva]))
            p = mm.predict(Xte)
            filas[nm] = met_reg(yte, p)
        except Exception as ex:
            print(f'  [!] {nm} fallo en test: {ex}')
    print('\n--- TEST ---')
    print(tabla(pd.DataFrame(filas).T.reset_index().rename(columns={'index': 'modelo'}),
                ['modelo', 'MAE', 'RMSE', 'R2', 'MAE_grave']))
    rt = filas[camp]
    print(f'\n>> Campeon {camp}: TEST MAE={rt["MAE"]:.2f} dias R2={rt["R2"]:.3f} MAE_graves={rt["MAE_grave"]:.2f}')
    try:
        m_refit = clone(MR[camp]).fit(pd.concat([Xtr, Xva]), np.concatenate([ytr, yva]))
        q = float(np.quantile(np.abs(yva - m_refit.predict(Xva)), min(1.0, (1 - ALPHA) * (1 + 1 / len(Xva)))))
        pr = m_refit.predict(Xte)
        lo, hi = pr - q, pr + q
        print(f'Conformal 80%: q={q:.2f}d cobertura test={np.mean((yte >= lo) & (yte <= hi)) * 100:.1f}%')
    except Exception as ex:
        print(f'  [!] conformal fallo: {ex}')
    acum['reg'].append(pd.DataFrame(filas).T)
    acum['camp'].append(dict(campeon=camp, **{k: rt[k] for k in ('MAE', 'RMSE', 'R2', 'MAE_grave')}))


# ----------------------------------------------------------------------------- PANEL + PROSPECTIVO + MAIN
def panel_riesgo(df, pred_grave, acum):
    d = df.copy()
    d['p_grave'] = pred_grave
    d['costo_esp'] = d['p_grave'] * DEMURRAGE * 8
    g = d.groupby(['puerto', 'tipo_dom']).agg(n=('p_grave', 'size'), p_grave=('p_grave', 'mean'),
                                              costo_esp=('costo_esp', 'sum')).round(2)
    g = g.sort_values('costo_esp', ascending=False).head(15).reset_index()
    print('\n--- PANEL DE RIESGO ruta x transportista (top costo esperado) ---')
    print(tabla(g))
    print(f'(supuesto: demurrage {DEMURRAGE:,.0f} USD/dia x 8 dias medios graves)')
    acum['panel'] = g


def main():
    t0 = time.time()
    print('Librerias:', {'numpy': np.__version__, 'pandas': pd.__version__,
                          'sklearn': __import__('sklearn').__version__})
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        sys.stdout = Tee(os.path.join(OUT_DIR, 'log_completo.txt'))
    except Exception:
        pass
    banner('PROYECTO 4 | RETRASOS EN ENVIOS Y EMBARQUES (demurrage)')
    be = Backend()
    df = preparar(be.cargar())
    n = len(df)
    a, b = int(n * FRAC_TRAIN), int(n * (FRAC_TRAIN + FRAC_VAL))
    tr, va, te = df.iloc[:a], df.iloc[a:b], df.iloc[b:]
    print(f'train={len(tr)} val={len(va)} test={len(te)} (temporal por FECHA_PROGRAMADA)')
    print(f'Clases test: {pd.Series(te["y_sev"]).value_counts(normalize=True).round(2).to_dict()}')
    acum = dict(clf=[], reg=[], camp=[], campclf=[], imp=[], panel=[])
    pred_test = evaluar_clf(tr, va, te, modelos_clf(), acum)
    evaluar_reg(tr, va, te, modelos_reg(), acum)
    # panel con tasa grave predicha (0/1 del campeon) por ruta x transportista
    te2 = te.copy()
    panel_riesgo(te2, (pred_test == 2).astype(float), acum)
    # prospectivo: contexto de los ultimos 500 embarques programados
    try:
        ult = df.tail(500)
        print(f'\nProspectivo: {len(ult)} ultimos embarques ({ult["fecha_programada"].min()} -> {ult["fecha_programada"].max()})')
        print(f'Graves observados ahi: {(ult["y_sev"].values == 2).mean():.1%} (referencia para vigilar rutas del panel)')
    except Exception as ex:
        print(f'  [!] prospectivo fallo: {ex}')
    cc = lambda k: pd.concat(acum[k], ignore_index=True) if acum[k] and isinstance(acum[k][0], pd.DataFrame) else pd.DataFrame(acum[k])
    df_clf, df_reg = cc('clf'), cc('reg')
    banner('RESUMEN_PARA_COMPARTIR (copia desde aqui)')
    L = ['[A] CLASIFICACION test', tabla(df_clf[df_clf.fase == 'test'].sort_values('F1_macro', ascending=False)),
         '\n[B] REGRESION test', tabla(acum['reg'][0].reset_index().rename(columns={'index': 'modelo'})),
         '\n[C] CAMPEONES', tabla(pd.DataFrame(acum['camp'] + acum['campclf'])),
         '\n[D] PANEL DE RIESGO top', tabla(acum['panel'])]
    print('\n'.join(L))
    print(f'\nTiempo total: {(time.time() - t0) / 60:.1f} min')
    print('\n' + '=' * 100 + '\nFIN RESUMEN_PARA_COMPARTIR\n' + '=' * 100)
    try:
        for nom, dd in [('metricas_clasificacion', df_clf),
                        ('campeones', pd.DataFrame(acum['camp'] + acum['campclf'])),
                        ('panel_riesgo', acum['panel'])]:
            if len(dd):
                dd.to_csv(os.path.join(OUT_DIR, f'{nom}.csv'), index=False, encoding='utf-8')
        print(f'\nArchivos guardados en: {os.path.abspath(OUT_DIR)}')
    except Exception as ex:
        print(f'[!] no se pudieron guardar archivos locales: {ex}')
    if ESCRIBIR:
        be.escribir(df_clf, 'RETRASOS_METRICAS_CLF')
        be.escribir(acum['panel'], 'RETRASOS_PANEL_RIESGO')
    if hasattr(sys.stdout, 'f'):
        sys.stdout = sys.stdout.o


if __name__ == '__main__':
    main()
