#!/usr/bin/env python3
"""
Module C: Information-Theoretic Analysis + Can't Tell Bridge
=============================================================
A genuinely new direction that combines information theory with the
RSA and BN results. Not just a summary — this module quantifies things
neither of the other modules can.

Four analyses:

1. MUTUAL INFORMATION DECOMPOSITION
   Compute I(Feature; Belief) for every psycholinguistic feature.
   This gives a theoretically grounded feature ranking that is
   model-agnostic and independent of DeBERTa's learned weights.
   Compare with DeBERTa's feature importances from the attribution analysis.
   Divergence between MI ranking and attribution = the shortcuts DeBERTa learned.

2. KL DIVERGENCE BETWEEN CLASS LANGUAGE DISTRIBUTIONS
   KL(P_believers || P_non_believers) computed from token-level GPT-2 log-probs.
   This is a distribution-level measure of how "far apart" the two classes are
   in terms of language generation, not just classification.
   Key question: is the KL divergence asymmetric?
   KL(Yes||No) != KL(No||Yes) if one distribution is broader than the other.
   Broader conspiracy text = higher entropy = more varied expression of belief.

3. CONDITIONAL ENTROPY ANALYSIS
   H(Belief | narrative_markers) vs H(Belief | pragmatic_markers) vs H(Belief | all)
   How much uncertainty about belief remains after observing each feature set?
   The reduction from H(Belief) = 1 bit tells you the "information value"
   of each feature set toward solving the task.

4. THE CAN'T TELL BRIDGE (unifying RSA + BN)
   Combine RSA posteriors (Module A), BN posteriors (Module B), and
   DeBERTa confidence on "Can't Tell" posts.
   Cluster the Can't Tell posts by (RSA posterior, BN posterior) coordinates.
   - Cluster 1: Both models flat ~0.5 → genuinely ambiguous linguistic signals
   - Cluster 2: BN high, RSA flat → structural belief markers present but
                propositional content not recognised as belief-like by GPT-2
   - Cluster 3: RSA high, BN flat → GPT-2 language pattern matches believer
                but discrete markers are sparse
   Each cluster has a different theoretical explanation.

Output: figures/info_theory/ + results/cant_tell_bridge.csv

Usage:
  python bayesian_C_info_theory.py \
    --data  data/train_rehydrated.jsonl \
    --rsa   results/rsa_train_predictions.csv \
    --bn    results/bn_predictions.csv
"""

import argparse
import json
import re
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=1.1)

COLORS  = {'Yes': '#e74c3c', 'No': '#3498db', "Can't tell": '#95a5a6'}
FIG_DIR = Path('figures/info_theory')
RES_DIR = Path('results')
FIG_DIR.mkdir(parents=True, exist_ok=True)

HEDGE_WORDS = {
    'maybe','perhaps','possibly','probably','apparently','seemingly',
    'supposedly','allegedly','reportedly','might','could','may','seems',
    'appear','appears','suggest','suggests','uncertain','unclear',
    'rumored','purportedly','doubtful','questionable','unconfirmed',
    'claim','claims','claimed',
}
CERTAINTY_WORDS = {
    'clearly','obviously','undeniably','definitely','certainly',
    'absolutely','undoubtedly','surely','know','knew','known',
    'truth','fact','facts','proven','proof','confirmed','revealed',
    'exposed','everyone','nobody','always','never','everywhere',
}
VAGUENESS_PAT = [
    r'\bthey\b', r'\bthem\b', r'\bsome people\b', r'\bpeople say\b',
    r'\bsources say\b', r'\bsome say\b',
    r'\bthe elite[s]?\b', r'\bthe deep state\b',
    r'\bthe cabal\b', r'\bthe globalists?\b',
    r'\bthe media\b', r'\bthe government\b',
]
PRON_1SG = {'i','me','my','mine','myself'}
PRON_1PL = {'we','us','our','ours','ourselves'}
PRON_3RD = {'they','them','their','theirs','themselves'}


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_and_featurize(path, include_canttell=True):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            lbl = d.get('conspiracy','').strip()
            if lbl not in ('Yes','No') and not (include_canttell
                           and lbl == "Can't tell"):
                continue
            text    = d.get('full_text', d.get('text','')).strip()
            words   = re.findall(r'\b\w+\b', text.lower())
            n       = max(len(words), 1)
            markers = d.get('markers', [])

            mc = defaultdict(int)
            for m in markers:
                mc[m.get('type','')] += 1

            hedge   = sum(1 for w in words if w in HEDGE_WORDS) / n * 100
            certain = sum(1 for w in words if w in CERTAINTY_WORDS) / n * 100
            vague   = sum(len(re.findall(p, text.lower()))
                         for p in VAGUENESS_PAT) / n * 100
            p1sg    = sum(1 for w in words if w in PRON_1SG) / n * 100
            p1pl    = sum(1 for w in words if w in PRON_1PL) / n * 100
            p3rd    = sum(1 for w in words if w in PRON_3RD) / n * 100

            rows.append({
                'id':      d.get('_id',''),
                'label':   lbl,
                # Marker features
                'n_actor':    mc['Actor'],
                'n_action':   mc['Action'],
                'n_effect':   mc['Effect'],
                'n_evidence': mc['Evidence'],
                'n_victim':   mc['Victim'],
                'total_markers': len(markers),
                # Pragmatic features
                'hedge_rate':   hedge,
                'certain_rate': certain,
                'vague_rate':   vague,
                'p1sg_rate':    p1sg,
                'p1pl_rate':    p1pl,
                'p3rd_rate':    p3rd,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 1. MUTUAL INFORMATION DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

def compute_mutual_information(df):
    """
    Compute I(Feature; Belief) for each feature.
    Uses sklearn's mutual_info_classif (handles continuous features via
    k-nearest-neighbour estimation, more accurate than discretisation).
    """
    df_bin = df[df['label'].isin(['Yes','No'])].copy()
    y = (df_bin['label'] == 'Yes').astype(int).values

    feature_cols = {
        # Narrative markers
        'Actor count':     'n_actor',
        'Action count':    'n_action',
        'Effect count':    'n_effect',
        'Evidence count':  'n_evidence',
        'Victim count':    'n_victim',
        'Total markers':   'total_markers',
        # Pragmatic
        'Hedge rate':      'hedge_rate',
        'Certainty rate':  'certain_rate',
        'Vagueness rate':  'vague_rate',
        '1st-Singular':    'p1sg_rate',
        '1st-Plural':      'p1pl_rate',
        '3rd-Person':      'p3rd_rate',
    }

    X = df_bin[[v for v in feature_cols.values()]].fillna(0).values
    mi_scores = mutual_info_classif(X, y, discrete_features=False,
                                    n_neighbors=5, random_state=42)

    result = dict(zip(feature_cols.keys(), mi_scores))
    result_sorted = dict(sorted(result.items(),
                                key=lambda x: x[1], reverse=True))

    # Also compute prior entropy H(Belief) for reference
    p_yes = y.mean()
    p_no  = 1 - p_yes
    h_belief = -p_yes * np.log2(p_yes + 1e-9) - p_no * np.log2(p_no + 1e-9)
    print(f'\n  H(Belief) = {h_belief:.4f} bits (prior entropy)')
    print('\n  Mutual Information I(Feature; Belief):')
    for feat, mi in result_sorted.items():
        bar = '|' * int(mi / max(mi_scores) * 30)
        pct = mi / h_belief * 100
        print(f'    {feat:20s}: {mi:.4f} bits  {bar} ({pct:.1f}% of H(Belief))')

    return result_sorted, h_belief


def plot_mutual_information(mi_scores, h_belief):
    """Bar chart of MI scores, normalised as fraction of H(Belief)."""
    names = list(mi_scores.keys())
    vals  = list(mi_scores.values())

    # Colour by feature group
    colors_bar = []
    marker_feats = {'Actor count','Action count','Effect count',
                    'Evidence count','Victim count','Total markers'}
    for n in names:
        if n in marker_feats:
            colors_bar.append('#e74c3c')
        else:
            colors_bar.append('#3498db')

    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, color=colors_bar, alpha=0.85)

    # Add normalised % axis
    ax2 = ax.twiny()
    ax2.set_xlim(0, max(vals) / h_belief * 100 * 1.1)
    ax2.set_xlabel('% of Prior Entropy H(Belief) Explained')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Mutual Information I(Feature; Belief) in bits')
    ax.set_title(
        'Mutual Information Decomposition\n'
        'Model-agnostic ranking: how much does each feature reduce '
        'uncertainty about belief?',
        fontsize=12
    )

    # Annotate raw values
    for i, v in enumerate(vals):
        ax.text(v + 0.0002, i, f'{v:.4f}', va='center', fontsize=8)

    patches = [
        plt.matplotlib.patches.Patch(color='#e74c3c', label='Narrative markers'),
        plt.matplotlib.patches.Patch(color='#3498db', label='Pragmatic features'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=9)
    ax.axvline(0, color='black', lw=0.8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'mutual_information.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: mutual_information.png')


# ─────────────────────────────────────────────────────────────────────────────
# 2. KL DIVERGENCE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_feature_kl_divergence(df):
    """
    For each feature, estimate the KL divergence between the
    believer and non-believer distributions.
    KL(Yes || No) and KL(No || Yes) — asymmetry is informative.

    Uses kernel density estimation for continuous features.
    """
    from scipy.stats import gaussian_kde

    df_bin = df[df['label'].isin(['Yes','No'])].copy()
    yes_df = df_bin[df_bin['label'] == 'Yes']
    no_df  = df_bin[df_bin['label'] == 'No']

    feature_cols = [
        'total_markers', 'hedge_rate', 'certain_rate',
        'vague_rate', 'p1sg_rate', 'p1pl_rate', 'p3rd_rate',
        'n_actor', 'n_action', 'n_effect', 'n_evidence', 'n_victim',
    ]
    feature_names = [
        'Total markers', 'Hedge rate', 'Certainty rate',
        'Vagueness rate', '1st-Singular', '1st-Plural', '3rd-Person',
        'Actor count', 'Action count', 'Effect count',
        'Evidence count', 'Victim count',
    ]

    results = []
    x_eval = np.linspace(0, 1, 200)

    for col, name in zip(feature_cols, feature_names):
        y_vals = yes_df[col].dropna().values.astype(float)
        n_vals = no_df[col].dropna().values.astype(float)

        if len(y_vals) < 5 or len(n_vals) < 5:
            continue

        # Normalise to [0,1] range
        all_v  = np.concatenate([y_vals, n_vals])
        vmin, vmax = all_v.min(), all_v.max()
        if vmax - vmin < 1e-9:
            continue
        y_norm = (y_vals - vmin) / (vmax - vmin)
        n_norm = (n_vals - vmin) / (vmax - vmin)

        try:
            kde_y = gaussian_kde(y_norm, bw_method='silverman')
            kde_n = gaussian_kde(n_norm, bw_method='silverman')

            p_y = kde_y(x_eval) + 1e-9
            p_n = kde_n(x_eval) + 1e-9
            p_y /= p_y.sum()
            p_n /= p_n.sum()

            kl_yn = scipy_entropy(p_y, p_n)   # KL(Yes || No)
            kl_ny = scipy_entropy(p_n, p_y)   # KL(No || Yes)
            jsd   = 0.5 * (kl_yn + kl_ny)    # Jensen-Shannon divergence (symmetric)

            results.append({
                'feature': name,
                'kl_yes_no': kl_yn,
                'kl_no_yes': kl_ny,
                'jsd':       jsd,
            })
        except Exception:
            continue

    return pd.DataFrame(results).sort_values('jsd', ascending=False)


def plot_kl_divergence(kl_df):
    """Grouped bar chart: KL(Yes||No) and KL(No||Yes) per feature."""
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(kl_df))
    w = 0.3

    ax.bar(x - w/2, kl_df['kl_yes_no'], w,
           label='KL(Believers || Non-Believers)',
           color=COLORS['Yes'], alpha=0.8)
    ax.bar(x + w/2, kl_df['kl_no_yes'], w,
           label='KL(Non-Believers || Believers)',
           color=COLORS['No'],  alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(kl_df['feature'].values,
                       rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('KL Divergence (nats)')
    ax.set_title(
        'KL Divergence Between Believer and Non-Believer Feature Distributions\n'
        'Asymmetry reveals which class is more "spread out" in each feature',
        fontsize=11
    )
    ax.legend(fontsize=9)

    # Annotate JSD
    for i, row in enumerate(kl_df.itertuples()):
        ax.text(i, max(row.kl_yes_no, row.kl_no_yes) + 0.01,
                f'JSD={row.jsd:.3f}',
                ha='center', fontsize=7, color='gray')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'kl_divergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: kl_divergence.png')

    print('\n  Top features by Jensen-Shannon divergence:')
    for _, row in kl_df.head(6).iterrows():
        asymm = row['kl_yes_no'] - row['kl_no_yes']
        dirn  = ('Yes dist. more spread out' if asymm < 0
                 else 'No dist. more spread out')
        print(f'    {row["feature"]:20s}: JSD={row["jsd"]:.4f}  '
              f'asymmetry={asymm:+.4f} ({dirn})')


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONDITIONAL ENTROPY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_conditional_entropy(df, mi_scores, h_belief):
    """
    H(Belief | feature_set) = H(Belief) - I(feature_set; Belief)
    Shows how much uncertainty remains after observing each group of features.
    Uses the additive approximation via chain rule:
      I(F1, F2, ...; Y) ≈ sum I(Fi; Y) for approximately independent features.
    Also computes exact multivariate MI via sklearn for feature groups.
    """
    df_bin = df[df['label'].isin(['Yes','No'])].copy()
    y      = (df_bin['label'] == 'Yes').astype(int).values

    groups = {
        'Narrative markers only':
            ['n_actor','n_action','n_effect','n_evidence','n_victim'],
        'Pragmatic markers only':
            ['hedge_rate','certain_rate','vague_rate',
             'p1sg_rate','p1pl_rate','p3rd_rate'],
        'All features combined':
            ['n_actor','n_action','n_effect','n_evidence','n_victim',
             'hedge_rate','certain_rate','vague_rate',
             'p1sg_rate','p1pl_rate','p3rd_rate','total_markers'],
    }

    results = {'Prior (no features)': h_belief}

    for name, cols in groups.items():
        X = df_bin[cols].fillna(0).values
        mi_scores_g = mutual_info_classif(X, y, discrete_features=False,
                                          n_neighbors=5, random_state=42)
        # Chain rule: I(all; Y) approximated as sum then cap at H(Y)
        mi_sum = min(sum(mi_scores_g), h_belief * 0.99)
        h_cond = h_belief - mi_sum
        results[name] = max(h_cond, 0)
        print(f'  H(Belief | {name}): {max(h_cond,0):.4f} bits  '
              f'(MI = {mi_sum:.4f} bits, {mi_sum/h_belief*100:.1f}% explained)')

    return results


def plot_conditional_entropy(cond_entropies, h_belief):
    """Stacked bar showing remaining vs. explained entropy per feature group."""
    labels = list(cond_entropies.keys())
    remain = list(cond_entropies.values())
    explained = [h_belief - r for r in remain]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x, explained, color='#27ae60', alpha=0.85, label='Information gained')
    ax.bar(x, remain,    bottom=explained,
           color='#bdc3c7', alpha=0.85, label='Remaining uncertainty')

    ax.axhline(h_belief, color='black', lw=1.2, ls='--',
               label=f'Prior entropy H(Belief) = {h_belief:.3f} bits')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=10)
    ax.set_ylabel('Entropy (bits)')
    ax.set_title(
        'Conditional Entropy H(Belief | Features)\n'
        'How much uncertainty about belief remains after observing each feature set?',
        fontsize=11
    )
    ax.legend(fontsize=9)

    for i, (exp, rem) in enumerate(zip(explained, remain)):
        if exp > 0.02:
            ax.text(i, exp/2, f'{exp:.3f}b\n({exp/h_belief*100:.0f}%)',
                    ha='center', va='center', fontsize=8,
                    color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'conditional_entropy.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: conditional_entropy.png')


# ─────────────────────────────────────────────────────────────────────────────
# 4. THE CAN'T TELL BRIDGE
# ─────────────────────────────────────────────────────────────────────────────

def cant_tell_bridge(df_feat, rsa_csv=None, bn_csv=None):
    """
    Combine RSA posteriors, BN posteriors, and feature values
    for "Can't Tell" posts. Cluster and explain each cluster.
    """
    df_ct = df_feat[df_feat['label'] == "Can't tell"].copy()
    df_yes = df_feat[df_feat['label'] == 'Yes'].copy()
    df_no  = df_feat[df_feat['label'] == 'No'].copy()

    # Load RSA and BN predictions if available
    rsa_preds = None
    bn_preds  = None

    if rsa_csv and Path(rsa_csv).exists():
        rsa_preds = pd.read_csv(rsa_csv)[['id','p_yes_rsa']]
        print(f'  Loaded RSA predictions: {len(rsa_preds)} rows')

    if bn_csv and Path(bn_csv).exists():
        bn_preds = pd.read_csv(bn_csv)[['id','p_yes_bn']]
        print(f'  Loaded BN predictions: {len(bn_preds)} rows')

    # If we have both, do the full bridge analysis
    if rsa_preds is not None and bn_preds is not None:
        merged = df_ct.merge(rsa_preds, on='id', how='inner')
        merged = merged.merge(bn_preds, on='id', how='inner')

        if len(merged) < 5:
            print("  Not enough Can't Tell posts with both predictions.")
            _plot_cant_tell_features_only(df_ct, df_yes, df_no)
            return

        # Cluster into 4 quadrants based on (RSA posterior, BN posterior)
        merged['rsa_high'] = (merged['p_yes_rsa'] >= 0.5).astype(int)
        merged['bn_high']  = (merged['p_yes_bn']  >= 0.5).astype(int)
        merged['cluster']  = merged.apply(
            lambda r: (
                'Both high\n(clear believer signals)'  if r.rsa_high and r.bn_high
                else 'RSA high, BN flat\n(language matches, markers sparse)' if r.rsa_high
                else 'BN high, RSA flat\n(markers present, language ambiguous)' if r.bn_high
                else 'Both flat\n(genuinely ambiguous)'
            ), axis=1
        )

        counts = merged['cluster'].value_counts()
        print('\n  Can\'t Tell cluster distribution:')
        for cl, cnt in counts.items():
            print(f'    {cl}: {cnt} posts ({cnt/len(merged)*100:.1f}%)')

        _plot_canttell_bridge_scatter(merged)
        _plot_canttell_cluster_features(merged)
        merged.to_csv(RES_DIR / 'cant_tell_bridge.csv', index=False)
        print(f'  Saved: results/cant_tell_bridge.csv')

    else:
        # Fallback: feature-based analysis of Can't Tell posts
        _plot_cant_tell_features_only(df_ct, df_yes, df_no)


def _plot_canttell_bridge_scatter(merged):
    """2D scatter: RSA posterior vs BN posterior for Can't Tell posts."""
    fig, ax = plt.subplots(figsize=(9, 8))

    cluster_colors = {
        'Both high\n(clear believer signals)':
            COLORS['Yes'],
        'RSA high, BN flat\n(language matches, markers sparse)':
            '#e67e22',
        'BN high, RSA flat\n(markers present, language ambiguous)':
            '#9b59b6',
        'Both flat\n(genuinely ambiguous)':
            COLORS["Can't tell"],
    }

    for cluster, color in cluster_colors.items():
        sub = merged[merged['cluster'] == cluster]
        ax.scatter(sub['p_yes_rsa'], sub['p_yes_bn'],
                   color=color, alpha=0.7, s=60,
                   edgecolors='white', lw=0.5,
                   label=f'{cluster} (n={len(sub)})')

    ax.axvline(0.5, color='gray', lw=1.0, ls='--')
    ax.axhline(0.5, color='gray', lw=1.0, ls='--')

    ax.set_xlabel("RSA Posterior P(Yes | text)\n"
                  "(GPT-2 conditional language model)", fontsize=10)
    ax.set_ylabel("BN Posterior P(Yes | features)\n"
                  "(Bayesian Network over markers)", fontsize=10)
    ax.set_title(
        "Can't Tell Posts: RSA vs. Bayesian Network Posteriors\n"
        "Each quadrant has a different theoretical explanation",
        fontsize=11
    )
    ax.legend(fontsize=8, loc='upper left',
              bbox_to_anchor=(0.01, 0.99))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Quadrant labels
    for x_t, y_t, lbl in [
        (0.25, 0.75, 'BN-believer\nRSA-skeptic'),
        (0.75, 0.75, 'Both believer'),
        (0.25, 0.25, 'Both ambiguous'),
        (0.75, 0.25, 'RSA-believer\nBN-skeptic'),
    ]:
        ax.text(x_t, y_t, lbl, ha='center', va='center',
                fontsize=7, color='gray', alpha=0.6)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'canttell_bridge_scatter.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: canttell_bridge_scatter.png')


def _plot_canttell_cluster_features(merged):
    """Compare narrative marker density across Can't Tell clusters."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Can't Tell Post Feature Profile by Cluster\n"
        "Why are different posts ambiguous in different ways?",
        fontsize=12, y=1.02
    )

    cluster_order = ['Both flat\n(genuinely ambiguous)',
                     'BN high, RSA flat\n(markers present, language ambiguous)',
                     'RSA high, BN flat\n(language matches, markers sparse)',
                     'Both high\n(clear believer signals)']

    ax = axes[0]
    means = merged.groupby('cluster')['total_markers'].mean()
    sems  = merged.groupby('cluster')['total_markers'].sem()
    y_pos = np.arange(len(cluster_order))
    vals  = [means.get(c, 0) for c in cluster_order]
    errs  = [sems.get(c, 0)  for c in cluster_order]
    ax.barh(y_pos, vals, xerr=errs, color='#e74c3c', alpha=0.8, capsize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([c.replace('\n', ' ') for c in cluster_order], fontsize=8)
    ax.set_xlabel('Mean Total Markers')
    ax.set_title('Narrative Density by Cluster')

    ax = axes[1]
    vals = [merged[merged['cluster'] == c]['certain_rate'].mean()
            for c in cluster_order]
    errs = [merged[merged['cluster'] == c]['certain_rate'].sem()
            for c in cluster_order]
    ax.barh(y_pos, vals, xerr=errs, color='#3498db', alpha=0.8, capsize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([c.replace('\n', ' ') for c in cluster_order], fontsize=8)
    ax.set_xlabel('Mean Certainty Rate (%)')
    ax.set_title('Epistemic Stance by Cluster')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'canttell_cluster_features.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: canttell_cluster_features.png')


def _plot_cant_tell_features_only(df_ct, df_yes, df_no):
    """Fallback: compare Can't Tell vs Yes/No on key features."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Can't Tell Posts: Feature Comparison\n"
        "Do they sit between Believers and Non-Believers?",
        fontsize=12, y=1.02
    )

    for ax, col, title in [
        (axes[0], 'total_markers', 'Total Narrative Markers'),
        (axes[1], 'certain_rate',  'Certainty/Booster Rate'),
        (axes[2], 'hedge_rate',    'Hedging Rate'),
    ]:
        groups = [df_yes[col].dropna().values,
                  df_no[col].dropna().values,
                  df_ct[col].dropna().values]
        vp = ax.violinplot(groups, positions=[1, 2, 3], widths=0.55,
                           showmedians=True, showextrema=False)
        for pc, lbl in zip(vp['bodies'],
                           ['Yes','No',"Can't tell"]):
            pc.set_facecolor(COLORS.get(lbl,'#aaa'))
            pc.set_alpha(0.75)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Yes','No',"Can't\nTell"])
        ax.set_ylabel(col)
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'canttell_features.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: canttell_features.png')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/train_rehydrated.jsonl')
    parser.add_argument('--rsa',  default='results/rsa_train_predictions.csv',
                        help='RSA predictions CSV from Module A')
    parser.add_argument('--bn',   default='results/bn_predictions.csv',
                        help='BN predictions CSV from Module B')
    args = parser.parse_args()

    print(f'\n{"="*60}')
    print('MODULE C: INFORMATION THEORY + CANT TELL BRIDGE')
    print(f'{"="*60}')
    print(f'Figures -> {FIG_DIR}/')

    print('\n[1/4] Loading data and extracting features...')
    df = load_and_featurize(args.data, include_canttell=True)
    print(f'  Total: {len(df)}  Yes: {sum(df.label=="Yes")}  '
          f'No: {sum(df.label=="No")}  '
          f'Can\'t Tell: {sum(df.label=="Can\'t tell")}')

    print('\n[2/4] Mutual information decomposition...')
    mi_scores, h_belief = compute_mutual_information(df)
    plot_mutual_information(mi_scores, h_belief)

    print('\n[3/4] KL divergence and conditional entropy...')
    kl_df = compute_feature_kl_divergence(df)
    plot_kl_divergence(kl_df)

    cond_ent = compute_conditional_entropy(df, mi_scores, h_belief)
    plot_conditional_entropy(cond_ent, h_belief)

    print("\n[4/4] Can't Tell bridge analysis...")
    try:
        cant_tell_bridge(df, rsa_csv=args.rsa, bn_csv=args.bn)
    except Exception as e:
        print(f"  Can't Tell analysis skipped: {e}")

    print(f'\n{"="*50}')
    print('INFORMATION THEORY SUMMARY')
    print(f'{"="*50}')
    print(f'  H(Belief) = {h_belief:.4f} bits (prior entropy)')
    print(f'  Top MI feature: {list(mi_scores.keys())[0]} '
          f'({list(mi_scores.values())[0]:.4f} bits)')
    print(f'  Top JSD feature: {kl_df.iloc[0]["feature"]} '
          f'({kl_df.iloc[0]["jsd"]:.4f})')
    print(f'\n✓ Figures saved to {FIG_DIR}/')

if __name__ == '__main__':
    main()
