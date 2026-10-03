#!/usr/bin/env python3
"""Mineria DESCRIPTIVA (sin objetivo): que tipos de entrega existen, que co-ocurre y que es raro.
Bronce cuenta nueva (ARQUI1.CAPA_BRONCE). k-means + jerarquico + Apriori propio + DBSCAN.
Uso:  python mineria_patrones_bronce.py   (SAMPLE_PCT=100; prueba con 2)
Guarda docs/img/patrones.png + imprime REPORTE.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
SEED = 42
def conectar():
    import snowflake.connector
    pwd = os.getenv('SNOWFLAKE_PASSWORD', '').strip().strip('\'"')
    if not pwd:
        try:
            import pathlib as _pl
            pwd = _pl.Path.home().joinpath('.snowflake_pwd').read_text(encoding='utf-8').strip()
        except Exception:
            pwd = ''
    if not pwd:
        raise SystemExit("Sin password: export SNOWFLAKE_PASSWORD='clave' o ~/.snowflake_pwd")
    kw = dict(user=os.getenv('SNOWFLAKE_USER', 'ENRIQUE'), password=pwd,
              account=os.getenv('SNOWFLAKE_ACCOUNT', 'WMNAMCT-LN60692'),
              database=os.getenv('SNOWFLAKE_DB', 'ARQUI1'),
              schema=os.getenv('SNOWFLAKE_SCHEMA', 'CAPA_BRONCE'),
              warehouse=os.getenv('SNOWFLAKE_WH', 'COMPUTE_WH'), login_timeout=60)
    if os.getenv('SNOWFLAKE_ROLE'):
        kw['role'] = os.getenv('SNOWFLAKE_ROLE')
    return snowflake.connector.connect(**kw)
def apriori(trans, min_sup=0.05, min_conf=0.5, max_len=3):
    """Apriori compacto: devuelve reglas (antec, consec, soporte, confianza, lift)."""
    from itertools import combinations
    n = len(trans)
    items = {}
    for t in trans:
        for i in t:
            items[i] = items.get(i, 0) + 1
    freq = {frozenset([i]): c / n for i, c in items.items() if c / n >= min_sup}
    k = 2
    cur = {s for s in freq}
    while cur and k <= max_len:
        cand = set()
        lst = list(cur)
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                u = lst[a] | lst[b]
                if len(u) == k:
                    cand.add(u)
        nxt = {}
        for cset in cand:
            c = sum(1 for t in trans if cset <= t)
            if c / n >= min_sup:
                nxt[cset] = c / n
        freq.update(nxt)
        cur = set(nxt)
        k += 1
    rules = []
    for s, sup in freq.items():
        if len(s) < 2:
            continue
        for r in range(1, len(s)):
            for ante in combinations(s, r):
                ante, consec = frozenset(ante), s - frozenset(ante)
                if ante in freq:
                    conf = sup / freq[ante]
                    if conf >= min_conf:
                        lift = conf / freq[consec]
                        rules.append((ante, consec, sup, conf, lift))
    return sorted(rules, key=lambda x: -x[4])
def main():
    sample = float(os.getenv('SAMPLE_PCT', '100'))
    s = '' if sample >= 100 else f' TABLESAMPLE SYSTEM ({sample:g})'
    c = conectar(); cur = c.cursor()
    cur.execute(f"""SELECT PESO_BRUTO_TN - TARA_TN AS PESO_NETO, HUMEDAD_PCT, IMPUREZAS_PCT,
        ACEITE_PCT, PROTEINA_PCT, TEMPERATURA_GRANO_C, CALIDAD_GRANO,
        EXTRACT(MONTH FROM FECHA_HORA) AS MES, EXTRACT(HOUR FROM FECHA_HORA) AS HORA, ID_SILO
        FROM RECEPCION_GRANO{s}
        WHERE PESO_BRUTO_TN - TARA_TN BETWEEN 1 AND 200 AND FECHA_HORA BETWEEN '2020-01-01' AND '2027-01-01'""")
    df = pd.DataFrame(cur.fetchall(), columns=[d[0].lower() for d in cur.description])
    for col in ['peso_neto', 'humedad_pct', 'impurezas_pct', 'aceite_pct', 'proteina_pct', 'temperatura_grano_c']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['peso_neto', 'humedad_pct', 'impurezas_pct']).copy()
    print(f'Entregas: {len(df)}')
    F = ['peso_neto', 'humedad_pct', 'impurezas_pct', 'aceite_pct', 'proteina_pct', 'temperatura_grano_c']
    nd = df.dropna(subset=['peso_neto', 'humedad_pct', 'impurezas_pct']).copy()
    print(f'Entregas base: {len(nd)} (aceite/proteina/temp se imputan con mediana)')
    for col in ['aceite_pct', 'proteina_pct', 'temperatura_grano_c']:
        nd[col] = pd.to_numeric(nd[col], errors='coerce').fillna(pd.to_numeric(nd[col], errors='coerce').median())
    df = nd
    Xs = StandardScaler().fit_transform(np.log1p(df[F].clip(lower=0)))
    KM = KMeans if len(df) < 200000 else MiniBatchKMeans
    kw = dict(n_init=20, random_state=SEED) if KM is KMeans else dict(batch_size=4096, n_init=5, random_state=SEED)
    print('== Tipos de entrega (k-means) ==')
    best, BK = -1, 3
    for k in (3, 4, 5):
        lab = KM(n_clusters=k, **kw).fit_predict(Xs)
        sil = silhouette_score(Xs[:30000], lab[:30000]) if len(Xs) > 30000 else silhouette_score(Xs, lab)
        print(f'K={k} sil={sil:.3f} sizes={pd.Series(lab).value_counts().sort_index().to_dict()}')
        if sil > best:
            best, BK = sil, k
    lab = KM(n_clusters=BK, **kw).fit_predict(Xs)
    df['tipo'] = lab
    print(df.groupby('tipo')[F].mean().round(2).to_string())
    j = AgglomerativeClustering(n_clusters=BK).fit(Xs[:20000]) if len(Xs) >= 20000 else None
    if j is not None:
        from sklearn.metrics import adjusted_rand_score
        print(f'Jerarquico vs k-means ARI={adjusted_rand_score(lab[:20000], j.labels_):.3f} (consistencia del patron)')
    db = DBSCAN(eps=1.0, min_samples=30).fit(Xs[:100000] if len(Xs) > 100000 else Xs)
    print(f'DBSCAN raros: {(db.labels_ == -1).sum()} ({(db.labels_ == -1).mean():.2%})')
    print('== Reglas de asociacion (discretizado) ==')
    dd = df.copy()
    dd['VOL'] = pd.qcut(dd['peso_neto'], 3, labels=['chico', 'medio', 'grande'], duplicates='drop')
    dd['HUM'] = pd.cut(dd['humedad_pct'], [-1, 12, 14, 100], labels=['seca', 'limite', 'humeda'])
    dd['IMP'] = pd.cut(dd['impurezas_pct'], [-1, 0.5, 1, 100], labels=['limpio', 'borde', 'sucio'])
    dd['CAL'] = dd['calidad_grano'].fillna('?').astype(str)
    trans = [set(x) for x in dd[['VOL', 'HUM', 'IMP', 'CAL']].astype(str).apply(lambda r: [f'{c}={v}' for c, v in r.items()], axis=1)]
    rules = apriori(trans, min_sup=0.05, min_conf=0.5)[:15]
    print(f'{len(rules)} reglas top por lift:')
    for a, b, sup, conf, lift in rules:
        print(f'  {set(a)} => {set(b)}  sup={sup:.3f} conf={conf:.3f} lift={lift:.2f}')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    df.groupby('tipo')['peso_neto'].mean().plot.bar(ax=ax[0]); ax[0].set_title(f'Peso medio por tipo (K={BK})')
    df.groupby('tipo')['humedad_pct'].mean().plot.bar(ax=ax[1]); ax[1].set_title('Humedad media por tipo')
    if rules:
        rr = pd.DataFrame([{'sup': r[2], 'conf': r[3], 'lift': r[4]} for r in rules])
        ax[2].scatter(rr['sup'], rr['conf'], s=rr['lift'] * 40, alpha=.6); ax[2].set_title('Reglas: soporte vs confianza (tam=lift)')
        ax[2].set_xlabel('soporte'); ax[2].set_ylabel('confianza')
    plt.tight_layout()
    os.makedirs('docs/img', exist_ok=True)
    plt.savefig('docs/img/patrones.png', dpi=110)
    print('docs/img/patrones.png OK')
    print('=== REPORTE: tipos + top reglas + % raros arriba ===')
if __name__ == '__main__':
    main()
