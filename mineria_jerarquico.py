#!/usr/bin/env python3
"""Clustering jerarquico multidominio (mercado x operacion x comercial, 60 meses).
Relaciones a priori inconexas + graficos impactantes (dendrograma, clustermap, PCA).
Uso: python mineria_jerarquico.py  (lee data/matriz_mensual_multidominio.csv)
Guarda docs/img/dendrograma.png, clustermap.png + data/meses_clusters.csv
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
sns.set_theme(style='white', palette='muted')
MK = ['soya', 'aceite', 'pet', 'nino']
OP = ['n_rec', 'vol', 'hum', 'imp', 'merma']
CO = ['vtas', 'precio', 'basis']
def main():
    df = pd.read_csv('data/matriz_mensual_multidominio.csv')
    feats = MK + OP + CO
    d = df.dropna(subset=feats).copy()
    print(f'Meses: {len(d)}')
    Xs = StandardScaler().fit_transform(d[feats])
    print('== Vinculacion (cofenetica, mayor = mejor jerarquia) ==')
    best, bl = -1, None
    for m in ('ward', 'complete', 'average'):
        Z = linkage(Xs, method=m)
        c, _ = cophenet(Z, pdist(Xs))
        print(f'{m}: cophenet={c:.3f}')
        if c > best:
            best, bl, bZ = c, m, Z
    print(f'Metodo: {bl}')
    for k in (3, 4):
        lab = fcluster(bZ, k, criterion='maxclust')
        print(f'k={k} sil={silhouette_score(Xs, lab):.3f} sizes={pd.Series(lab).value_counts().sort_index().to_dict()}')
    d['cluster'] = fcluster(bZ, 3, criterion='maxclust')
    print(d.groupby('cluster')[feats].mean().round(2).to_string())
    print('== Top correlaciones CRUZADAS (mercado vs operacion/comercial) ==')
    C = pd.DataFrame(Xs, columns=feats).corr()
    cross = []
    for a in MK:
        for b in OP + CO:
            cross.append((abs(C.loc[a, b]), a, b, C.loc[a, b]))
    for v, a, b, s in sorted(cross, reverse=True)[:8]:
        print(f'{a} x {b}: r={s:+.3f}')
    import os
    os.makedirs('docs/img', exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(bZ, labels=d['mes'].values, leaf_rotation=90, leaf_font_size=6, ax=ax)
    ax.set_title('Dendrograma multidominio (Ward): meses que se comportan igual')
    plt.tight_layout(); plt.savefig('docs/img/dendrograma.png', dpi=110); plt.close()
    g = sns.clustermap(pd.DataFrame(Xs, columns=feats), method=bl, cmap='coolwarm', figsize=(10, 12),
                       yticklabels=d['mes'].values)
    g.fig.suptitle('Clustermap: que meses y que variables se mueven juntas', y=0.99)
    g.savefig('docs/img/clustermap.png', dpi=110); plt.close()
    d[['mes', 'cluster']].to_csv('data/meses_clusters.csv', index=False, encoding='utf-8')
    print('Guardados docs/img/dendrograma.png, clustermap.png, data/meses_clusters.csv')
if __name__ == '__main__':
    main()
