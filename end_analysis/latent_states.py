#!/usr/bin/env python3
"""
Experiment 1: Latent Epistemic States
========================================
The Bayesian Network posterior (Module B) showed the Yes class is bimodal:
mass clustered both near 0 AND near 0.8. This is direct evidence of
heterogeneity that binary classification collapses. There are sub-types
of believers (and non-believers) that have different linguistic profiles.

Method: Bayesian Gaussian Mixture Model with Dirichlet Process prior.
  - Fit on all posts in the 12-dimensional psycholinguistic feature space.
  - The DP prior determines the effective number of clusters automatically.
  - Map discovered clusters back to Yes/No/Can't Tell labels post-hoc.
  - Visualise with t-SNE in 2D.

What this will find (predictions based on the data):
  - 4-6 distinct profiles, not 2.
  - "Can't Tell" posts concentrate in specific intermediate clusters.
  - False positives (No posts misclassified as Yes) share a cluster with
    a subset of Yes posts -- explains the attribution/quoting problem.
  - Some believers score as "non-believers" linguistically -- hedged believers.

Usage:
  python exp1_latent_states.py --data data/train_rehydrated.jsonl
  python exp1_latent_states.py --data data/train_rehydrated.jsonl \
                               --bn-preds results/bn_predictions.csv
"""

import argparse, json, re, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from tqdm import tqdm

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=1.05)

OUT = Path('figures/exp1_latent'); OUT.mkdir(parents=True, exist_ok=True)
RES = Path('results');             RES.mkdir(parents=True, exist_ok=True)

LABEL_C = {'Yes':'#e74c3c', 'No':'#3498db', "Can't tell":'#95a5a6'}
CLUSTER_C = ['#e74c3c','#3498db','#27ae60','#f39c12',
              '#8e44ad','#16a085','#d35400','#2c3e50']

HEDGE  = {'maybe','perhaps','possibly','probably','apparently','seemingly',
          'supposedly','allegedly','reportedly','might','could','may',
          'seems','appear','appears','suggest','suggests','uncertain',
          'unclear','rumored','doubtful','questionable','claim','claims'}
CERT   = {'clearly','obviously','undeniably','definitely','certainly',
          'absolutely','undoubtedly','surely','know','knew','known',
          'truth','fact','facts','proven','proof','confirmed','revealed',
          'exposed','everyone','nobody','always','never','wake','woke'}
VAGUE  = [r'\bthey\b',r'\bthem\b',r'\bsome people\b',r'\bpeople say\b',
          r'\bsources say\b',r'\bsome say\b',r'\bthe elite[s]?\b',
          r'\bthe deep state\b',r'\bthe cabal\b',r'\bthe government\b']
MORAL  = {'evil','corrupt','betray','lie','lies','deceive','deception',
          'agenda','criminal','murder','killed','tyranny','oppression',
          'resist','fight','freedom','hoax','scam','trafficking',
          'fear','hate','anger','outrage','shocking','horrifying'}


# ── DATA ─────────────────────────────────────────────────────────────────────
def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            d   = json.loads(line.strip())
            lbl = d.get('conspiracy','').strip()
            txt = d.get('full_text', d.get('text','')).strip()
            if not txt: continue
            w   = re.findall(r'\b\w+\b', txt.lower()); n = max(len(w),1)
            mc  = defaultdict(int)
            for m in d.get('markers',[]):
                mc[m.get('type','')] += 1
            rows.append({
                'id':    d.get('_id',''),
                'label': lbl,
                'text':  txt[:300],
                'n_actor':    mc['Actor'],
                'n_action':   mc['Action'],
                'n_effect':   mc['Effect'],
                'n_evidence': mc['Evidence'],
                'n_victim':   mc['Victim'],
                'total_m':    len(d.get('markers',[])),
                'hedge':   sum(1 for x in w if x in HEDGE)/n*100,
                'cert':    sum(1 for x in w if x in CERT)/n*100,
                'vague':   sum(len(re.findall(p,txt.lower())) for p in VAGUE)/n*100,
                'p1sg':    sum(1 for x in w if x in {'i','me','my','mine','myself'})/n*100,
                'p1pl':    sum(1 for x in w if x in {'we','us','our','ours'})/n*100,
                'p3rd':    sum(1 for x in w if x in {'they','them','their','theirs'})/n*100,
                'moral':   sum(1 for x in w if x in MORAL)/n*100,
                'log_wc':  np.log1p(n),
            })
    return pd.DataFrame(rows)

FCOLS = ['n_actor','n_action','n_effect','n_evidence','n_victim',
         'total_m','hedge','cert','vague','p1sg','p1pl','p3rd','moral','log_wc']
FNAMES = ['Actor','Action','Effect','Evidence','Victim','Total markers',
          'Hedge rate','Certainty','Vagueness','1st-Sing','1st-Plur',
          '3rd-Person','Moral/Emot','Log words']


# ── FIT MODELS ────────────────────────────────────────────────────────────────
def fit_all(X_sc):
    print('  Fitting Dirichlet Process GMM (max_components=12)...')
    dp = BayesianGaussianMixture(
        n_components=12, covariance_type='full',
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=0.1,
        max_iter=600, n_init=5, random_state=42, tol=1e-4)
    dp.fit(X_sc)
    eff_k = int(np.sum(dp.weights_ > 1/12))
    print(f'  Effective clusters (DP-GMM): {eff_k}')

    print(f'  Fitting standard GMM for BIC at K=2..{eff_k+3}...')
    bics = {}
    best_gmm, best_bic, best_k = None, np.inf, 2
    for k in range(2, eff_k+4):
        g = GaussianMixture(n_components=k, covariance_type='full',
                             n_init=5, random_state=42, max_iter=400)
        g.fit(X_sc)
        bics[k] = g.bic(X_sc)
        if bics[k] < best_bic:
            best_bic, best_gmm, best_k = bics[k], g, k
    print(f'  BIC-optimal K = {best_k}')
    return dp, best_gmm, bics, eff_k, best_k


# ── PROFILE EACH CLUSTER ──────────────────────────────────────────────────────
THEORY = {
    'Propagandist':       (['cert','vague','total_m'], ['hedge','p1sg']),
    'Paranoid Victim':    (['n_victim','n_effect','moral','p3rd'], ['n_evidence','p1sg']),
    'Pseudo-Researcher':  (['n_evidence','n_actor','total_m'], ['moral','hedge']),
    'Critical Analyst':   (['p1sg','hedge'], ['total_m','cert','vague']),
    'Casual Observer':    (['log_wc'], ['total_m','cert']),
    'Hedged Believer':    (['hedge','n_actor','n_action'], ['cert','vague']),
}

def name_cluster(row):
    feat_vals = {c: row[c] for c in FCOLS if c in row}
    ranked    = sorted(feat_vals, key=feat_vals.get, reverse=True)
    top3, bot3 = set(ranked[:3]), set(ranked[-3:])
    scores = {}
    for tname,(hi,lo) in THEORY.items():
        scores[tname] = sum(1 for f in hi if f in top3) + \
                        sum(1 for f in lo if f in bot3)
    return max(scores, key=scores.get)

def profile(df, clusters):
    df = df.copy(); df['cluster'] = clusters
    rows = []
    for c in sorted(df['cluster'].unique()):
        s = df[df['cluster']==c]
        row = {'cluster':c, 'n':len(s),
               'pct_yes': (s.label=='Yes').mean()*100,
               'pct_no':  (s.label=='No').mean()*100,
               'pct_ct':  (s.label=="Can't tell").mean()*100}
        for f in FCOLS: row[f] = s[f].mean()
        rows.append(row)
    prof = pd.DataFrame(rows)
    prof['name'] = prof.apply(name_cluster, axis=1)
    # Deduplicate names
    seen = {}
    for i,r in prof.iterrows():
        n = r['name']
        if n in seen: seen[n]+=1; prof.at[i,'name'] = f"{n} ({seen[n]})"
        else: seen[n]=0
    return prof


# ── PLOTS ─────────────────────────────────────────────────────────────────────
def plot_bic(bics):
    fig, ax = plt.subplots(figsize=(8,4))
    ks,bs = zip(*sorted(bics.items()))
    ax.plot(ks, bs, 'o-', color='#2980b9', lw=2.5, ms=8)
    bk = ks[np.argmin(bs)]
    ax.axvline(bk, color='#e74c3c', ls='--', lw=1.5, label=f'Optimal K={bk}')
    ax.set_xlabel('K'); ax.set_ylabel('BIC')
    ax.set_title('BIC Model Selection — Number of Latent Epistemic Profiles', fontsize=11)
    ax.legend(); plt.tight_layout()
    plt.savefig(OUT/'bic_curve.png', dpi=150, bbox_inches='tight'); plt.close()
    print('  Saved: bic_curve.png')


def plot_composition(df, prof):
    fig, ax = plt.subplots(figsize=(max(10,len(prof)*1.8), 6))
    order = prof.sort_values('n', ascending=False)['name'].tolist()
    name_map = dict(zip(prof['cluster'], prof['name']))
    df2 = df.copy(); df2['cname'] = df2['cluster'].map(name_map)

    yes_p,no_p,ct_p,ns = [],[],[],[]
    for cn in order:
        s = df2[df2['cname']==cn]
        ns.append(len(s))
        yes_p.append((s.label=='Yes').mean()*100)
        no_p.append( (s.label=='No').mean()*100)
        ct_p.append( (s.label=="Can't tell").mean()*100)

    x = np.arange(len(order))
    ax.bar(x, yes_p, color='#e74c3c', alpha=0.85, label='Believers (Yes)')
    ax.bar(x, no_p,  bottom=yes_p, color='#3498db', alpha=0.85, label='Non-Believers (No)')
    cum = [a+b for a,b in zip(yes_p,no_p)]
    ax.bar(x, ct_p, bottom=cum, color='#95a5a6', alpha=0.85, label="Can't Tell")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('% of cluster'); ax.set_ylim(0,115)
    ax.set_title('Latent Epistemic Profile — Label Composition\n'
                 'Binary classification collapses these distinct profiles into Yes/No',
                 fontsize=12)
    ax.legend(loc='upper right')
    for i,n in enumerate(ns):
        ax.text(i, 106, f'n={n}', ha='center', fontsize=8, color='gray')
    plt.tight_layout()
    plt.savefig(OUT/'cluster_composition.png', dpi=150, bbox_inches='tight'); plt.close()
    print('  Saved: cluster_composition.png')


def plot_feature_heatmap(prof):
    readable = {'n_actor':'Actor','n_action':'Action','n_effect':'Effect',
                'n_evidence':'Evidence','n_victim':'Victim','total_m':'Total markers',
                'hedge':'Hedge rate','cert':'Certainty','vague':'Vagueness',
                'p1sg':'1st-Singular','p1pl':'1st-Plural','p3rd':'3rd-Person',
                'moral':'Moral/Emotion','log_wc':'Log word count'}
    mat = prof.set_index('name')[FCOLS].rename(columns=readable)
    # Z-score each column (within the cluster profiles)
    mat_z = (mat - mat.mean()) / (mat.std() + 1e-9)

    fig, ax = plt.subplots(figsize=(14, max(5,len(prof)*0.8)))
    sns.heatmap(mat_z, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                linewidths=0.4, ax=ax, annot_kws={'size':8})
    ax.set_title('Latent Profile Feature Heatmap (z-scored)\n'
                 'Red = above average, Blue = below average', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT/'feature_heatmap.png', dpi=150, bbox_inches='tight'); plt.close()
    print('  Saved: feature_heatmap.png')


def plot_tsne(df, X_sc, prof):
    print('  Running t-SNE (2D projection)...')
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, init='pca')
    coords = tsne.fit_transform(X_sc)
    df2 = df.copy()
    df2['tx'], df2['ty'] = coords[:,0], coords[:,1]
    name_map = dict(zip(prof['cluster'], prof['name']))
    df2['cname'] = df2['cluster'].map(name_map)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('t-SNE Projection of Psycholinguistic Feature Space', fontsize=13)

    # Left: coloured by cluster
    ax = axes[0]
    for i, (cn, grp) in enumerate(df2.groupby('cname')):
        ax.scatter(grp.tx, grp.ty, alpha=0.35, s=10,
                   color=CLUSTER_C[i % len(CLUSTER_C)], edgecolors='none', label=cn)
    ax.legend(fontsize=7, markerscale=3, loc='best')
    ax.set_title('Coloured by Discovered Cluster'); ax.axis('off')

    # Right: coloured by true label
    ax = axes[1]
    for lbl, grp in df2.groupby('label'):
        ax.scatter(grp.tx, grp.ty, alpha=0.3, s=10,
                   color=LABEL_C.get(lbl,'#aaa'), edgecolors='none', label=lbl)
    ax.legend(fontsize=9, markerscale=3)
    ax.set_title('Coloured by True Label'); ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUT/'tsne.png', dpi=150, bbox_inches='tight'); plt.close()
    print('  Saved: tsne.png')


def plot_canttell_cluster(df, prof):
    """Which clusters do 'Can't Tell' posts fall into?"""
    ct = df[df['label']=="Can't tell"]
    if len(ct) < 5:
        print("  Not enough Can't Tell posts."); return

    name_map  = dict(zip(prof['cluster'], prof['name']))
    ct2 = ct.copy(); ct2['cname'] = ct2['cluster'].map(name_map)
    counts = ct2['cname'].value_counts()

    # Compare to expected if uniformly distributed
    expected = len(ct) / len(prof)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(counts))
    bars = ax.bar(x, counts.values, color='#95a5a6', alpha=0.85)
    ax.axhline(expected, color='#e74c3c', lw=1.5, ls='--',
               label=f'Expected if uniform ({expected:.0f})')
    ax.set_xticks(x); ax.set_xticklabels(counts.index, rotation=20, ha='right')
    ax.set_ylabel("Count of Can't Tell posts"); ax.legend()
    ax.set_title("Which Latent Profile Do 'Can't Tell' Posts Fall Into?\n"
                 "Concentration above expected = genuine linguistic ambiguity in that cluster",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT/'canttell_clusters.png', dpi=150, bbox_inches='tight'); plt.close()
    print('  Saved: canttell_clusters.png')


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/train_rehydrated.jsonl')
    ap.add_argument('--bn-preds', default=None)
    args = ap.parse_args()

    print(f'\n{"="*60}\nEXPERIMENT 1: LATENT EPISTEMIC STATES\n{"="*60}')

    df  = load(args.data)
    X   = df[FCOLS].fillna(0).values
    sc  = StandardScaler(); X_sc = sc.fit_transform(X)
    n_ct = (df['label'] == "Can't tell").sum()
    print(f'  Loaded {len(df)} posts  '
          f'(Yes={sum(df.label=="Yes")}, No={sum(df.label=="No")}, '
          f'CT={n_ct}')

    dp, gmm, bics, eff_k, best_k = fit_all(X_sc)

    # Use BIC-optimal standard GMM clusters (more interpretable than DP)
    clusters = gmm.predict(X_sc)
    df['cluster'] = clusters

    # Silhouette score
    sil = silhouette_score(X_sc, clusters, sample_size=2000, random_state=42)
    print(f'  Silhouette score (K={best_k}): {sil:.4f}')

    # If BN preds available, compare
    # Calculate ARI (clusters vs true labels) before merging
    ari = adjusted_rand_score((df['label']=='Yes').astype(int).values, clusters)
    print(f'  ARI (clusters vs Yes/No): {ari:.4f}')

    # If BN preds available, merge them safely by dropping duplicates
    if args.bn_preds and Path(args.bn_preds).exists():
        bn = pd.read_csv(args.bn_preds)[['id','p_yes_bn']].drop_duplicates(subset=['id'])
        df = df.merge(bn, on='id', how='left')

    prof = profile(df, clusters)
    print('\n  Discovered profiles:')
    for _,r in prof.iterrows():
        print(f"  [{r['name']}] n={r['n']}  "
              f"Yes={r['pct_yes']:.0f}%  No={r['pct_no']:.0f}%  "
              f"CT={r['pct_ct']:.0f}%")

    df.to_csv(RES/'latent_states.csv', index=False)
    prof.to_csv(RES/'latent_profiles.csv', index=False)

    print('\nGenerating figures...')
    plot_bic(bics)
    plot_composition(df, prof)
    plot_feature_heatmap(prof)
    plot_tsne(df, X_sc, prof)
    plot_canttell_cluster(df, prof)

    print(f'\n{"="*50}\nSUMMARY')
    print(f'  BIC-optimal K: {best_k}  (Silhouette: {sil:.4f})')
    print(f'  Figures -> {OUT}/  |  Results -> {RES}/')

if __name__ == '__main__':
    main()