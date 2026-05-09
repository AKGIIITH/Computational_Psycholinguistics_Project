#!/usr/bin/env python3
"""
Experiment 2: Parameterized RSA — Alpha (Rationality) Estimation
==================================================================
Goodman & Frank (2016) RSA: the Pragmatic Speaker S1 chooses utterance u
proportional to exp(alpha * U(u,w)), where:
  U(u,w) = log L0(w|u) - C(u)   (utility = informativeness - cost)
  alpha   = rationality parameter (how hard the speaker is trying)

This module estimates alpha for each author using their own utterance as
the evidence. The core insight:
  - Conspiracy believers have a prior P(world) wildly divergent from their
    listeners' prior. To update the listener's belief, they must incur
    massive linguistic COST (dense narratives, heavy boosters, complex
    causal chains) while maximizing UTILITY (making the listener believe).
  - This predicts believers should have HIGHER alpha — they are more
    "optimised" or "desperate" speakers.
  - Non-believers discussing conspiracies have LOW alpha — they are not
    trying to update the listener's world model, just reporting.

Implementation:
  We operationalise the three RSA components from observable quantities:
  
  L0(belief | text): Naive literal interpretation = RSA posterior from Module A
                     (GPT-2 conditional perplexity). If not available, we
                     estimate it from the BN posterior.
  
  C(u): Production cost of the utterance. We estimate this as:
        C(u) = beta1 * narrative_density + beta2 * syntactic_complexity
               + beta3 * booster_density
        All are already computed. Higher marker count + deeper dep tree +
        more boosters = higher production cost.
  
  U(u,w): Utility = log(L0 | believer prior) - C(u)
  
  Alpha estimation: Given U(u,w), recover alpha by inverting the softmax:
        alpha = log(P_speaker / (1 - P_speaker)) / U(u,w)
        where P_speaker is the RSA/BN posterior.

This gives one alpha value per post. We compare:
  - alpha(Yes) vs alpha(No): believers should be higher
  - alpha distribution vs narrative density: more markers = higher rationality?
  - alpha for "Can't Tell" posts: should be intermediate

Additional: Cost decomposition
  - Which cost components matter most?
  - Is narrative density or pragmatic stance the bigger driver of cost?

Output: figures/exp2_alpha/
Usage:
  python exp2_alpha_rsa.py --data  data/train_rehydrated.jsonl \
                           --rsa   results/rsa_train_predictions.csv \
                           --bn    results/bn_predictions.csv
"""

import argparse, json, re, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import expit, logit  # sigmoid, logit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=1.1)

OUT = Path('figures/exp2_alpha'); OUT.mkdir(parents=True, exist_ok=True)
RES = Path('results');             RES.mkdir(parents=True, exist_ok=True)

C = {'Yes':'#e74c3c', 'No':'#3498db', "Can't tell":'#95a5a6'}

HEDGE = {'maybe','perhaps','possibly','probably','apparently','seemingly',
         'supposedly','allegedly','reportedly','might','could','may',
         'seems','appear','appears','suggest','suggests','uncertain',
         'unclear','rumored','doubtful','questionable','claim','claims'}
CERT  = {'clearly','obviously','undeniably','definitely','certainly',
         'absolutely','undoubtedly','surely','know','knew','known',
         'truth','fact','facts','proven','proof','confirmed','revealed',
         'exposed','everyone','nobody','always','never','wake','woke'}


# ── DATA ─────────────────────────────────────────────────────────────────────
def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            d   = json.loads(line.strip())
            lbl = d.get('conspiracy','').strip()
            txt = d.get('full_text', d.get('text','')).strip()
            if not txt: continue
            w  = re.findall(r'\b\w+\b', txt.lower()); n = max(len(w),1)
            mc = defaultdict(int)
            for m in d.get('markers',[]): mc[m.get('type','')] += 1
            rows.append({
                'id':       d.get('_id',''),
                'label':    lbl,
                'total_m':  len(d.get('markers',[])),
                'n_evidence': mc['Evidence'],
                'n_actor':    mc['Actor'],
                'n_victim':   mc['Victim'],
                'cert_rate':  sum(1 for x in w if x in CERT)/n*100,
                'hedge_rate': sum(1 for x in w if x in HEDGE)/n*100,
                'log_wc':     np.log1p(n),
            })
    return pd.DataFrame(rows)


# ── COST FUNCTION ─────────────────────────────────────────────────────────────
def compute_cost(df, beta=(0.4, 0.3, 0.3)):
    """
    C(u) = beta1*narrative_density + beta2*cert_rate + beta3*log_wc
    
    Narrative density = total marker count (normalised).
    Certainty rate    = production effort to assert strong claims.
    Log word count    = raw length cost.
    
    Weights beta are theory-motivated:
      - Narrative density is the most costly (constructing a 5-role frame
        requires deliberate compositional effort)
      - Certainty rate: expressing certainty without evidence requires
        cognitive effort to maintain internally contradictory stance
      - Word count: baseline length cost
    """
    sc = StandardScaler()
    feats = df[['total_m','cert_rate','log_wc']].fillna(0).values
    feats_sc = sc.fit_transform(feats)
    cost = feats_sc @ np.array(beta)
    # Shift to non-negative (cost must be >= 0)
    cost = cost - cost.min() + 0.01
    return cost


# ── ALPHA ESTIMATION ─────────────────────────────────────────────────────────
def estimate_alpha(posterior_p, cost, min_cost=0.01, clip_alpha=20.0):
    """
    Given:
      posterior_p = L0(belief | utterance) -- probability from RSA or BN model
      cost        = C(u) -- production cost of the utterance
    
    The RSA speaker chooses utterances proportional to exp(alpha * U).
    For a two-alternative choice:
      P_speaker(utterance | belief) = softmax(alpha * U)[belief]
    
    Inverting: alpha = logit(P_speaker) / U
    where U = log(P_speaker) - C(u)   (treating log-prob as informativeness proxy)
    
    We estimate U as: informativeness(u) - C(u)
      informativeness = the log posterior (how much the utterance updates belief)
    
    Note: alpha is only well-defined when P is not 0 or 1 and cost > 0.
    """
    p    = np.clip(posterior_p, 0.05, 0.95)   # avoid infinite logit
    c    = np.clip(cost, min_cost, None)

    # Informativeness proxy: log(p / (1-p)) -- how strongly does the utterance
    # distinguish the two belief states?
    informativeness = np.log(p / (1.0 - p))   # logit of posterior

    # Utility = informativeness - cost
    utility = informativeness - c

    # Alpha: scale factor. When utility > 0, speaker "gets value" from utterance.
    # alpha = logit(P) / utility (solving the softmax equation)
    # We use a stable formulation: alpha = informativeness / (utility + epsilon)
    epsilon = 0.1
    alpha   = informativeness / (utility + epsilon)
    alpha   = np.clip(alpha, -clip_alpha, clip_alpha)

    return alpha, informativeness, utility


# ── MAIN ANALYSIS ─────────────────────────────────────────────────────────────
def run_alpha_analysis(df):
    """
    Core analysis:
    1. Compute cost C(u) for every post.
    2. Use BN posterior as L0 (if available), else use heuristic from features.
    3. Estimate alpha per post.
    4. Compare alpha(Yes) vs alpha(No).
    5. Correlate alpha with narrative density, cert rate, hedge rate.
    """
    # Heuristic posterior if neither RSA nor BN preds are loaded:
    # Use narrative density + cert rate as a proxy for the speaker's success
    # at updating the listener (more markers + more certainty = higher posterior).
    df = df.copy()
    if 'p_posterior' not in df.columns:
        # Logistic mapping of total markers to a [0.1, 0.9] pseudo-posterior
        from scipy.special import expit
        raw = df['total_m'] * 0.3 + df['cert_rate'] * 0.5
        df['p_posterior'] = expit(raw - raw.mean())
        print('  Using feature-based posterior proxy (no RSA/BN preds loaded)')

    cost  = compute_cost(df)
    alpha, inform, utility = estimate_alpha(df['p_posterior'].values, cost)

    df['cost']        = cost
    df['inform']      = inform
    df['utility']     = utility
    df['alpha']       = alpha

    return df


# ── PLOTS ─────────────────────────────────────────────────────────────────────
def plot_alpha_distributions(df):
    """Violin: alpha distribution per label class."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('RSA Rationality Parameter Alpha (alpha) by Author Class\n'
                 'Higher alpha = speaker more optimised to update listener belief',
                 fontsize=13, y=1.02)

    for ax, col, title, ylabel in [
        (axes[0], 'alpha',   'Rationality Parameter alpha',       'alpha'),
        (axes[1], 'cost',    'Production Cost C(u)',               'Cost'),
        (axes[2], 'utility', 'Utility U = Informativeness - Cost', 'Utility'),
    ]:
        groups = [df[df['label']==l][col].dropna().values
                  for l in ('Yes','No') if l in df['label'].values]
        lbls   = [l for l in ('Yes','No') if l in df['label'].values]
        if not groups: continue

        vp = ax.violinplot(groups, positions=range(1,len(groups)+1),
                           widths=0.55, showmedians=True, showextrema=False)
        for pc, lbl in zip(vp['bodies'], lbls):
            pc.set_facecolor(C[lbl]); pc.set_alpha(0.75)
        ax.set_xticks(range(1,len(groups)+1))
        ax.set_xticklabels([f'Believers\n({l})' if l=='Yes' else
                            f'Non-Believers\n({l})' for l in lbls])
        ax.set_ylabel(ylabel); ax.set_title(title)

        # Stats
        if len(groups) == 2:
            _, p = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            star = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
            ax.text(0.97, 0.96, f'p={p:.4f} {star}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
            print(f'  {title}: Yes median={np.median(groups[0]):.3f}  '
                  f'No median={np.median(groups[1]):.3f}  p={p:.4f} {star}')

    plt.tight_layout()
    plt.savefig(OUT/'alpha_distributions.png', dpi=150, bbox_inches='tight')
    plt.close(); print('  Saved: alpha_distributions.png')


def plot_alpha_vs_density(df):
    """Scatter: alpha vs narrative density, coloured by label."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Alpha vs. Narrative Density\n'
                 'Do denser conspiracy narratives reflect higher speaker rationality?',
                 fontsize=12)

    for ax, col, xlabel in [
        (axes[0], 'total_m',   'Narrative Marker Count'),
        (axes[1], 'cert_rate', 'Certainty Booster Rate'),
    ]:
        for lbl in ('Yes','No'):
            sub = df[df['label']==lbl].dropna(subset=[col,'alpha'])
            ax.scatter(sub[col], sub['alpha'], alpha=0.2, s=12,
                       color=C[lbl], edgecolors='none', label=lbl)
            if len(sub) > 5:
                m,b,r,p,_ = stats.linregress(sub[col], sub['alpha'])
                xs = np.array([sub[col].min(), sub[col].max()])
                ax.plot(xs, m*xs+b, '--', color=C[lbl], lw=2,
                        label=f'{lbl} r={r:.3f}')
        ax.axhline(0, color='black', lw=0.8, ls=':')
        ax.set_xlabel(xlabel); ax.set_ylabel('alpha')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT/'alpha_vs_density.png', dpi=150, bbox_inches='tight')
    plt.close(); print('  Saved: alpha_vs_density.png')


def plot_cost_decomposition(df):
    """
    Show how each cost component contributes to alpha.
    This is the "linguistic cost" story: what are believers spending their
    effort on?
    """
    df_bin = df[df['label'].isin(['Yes','No'])].copy()

    components = {'Narrative markers': 'total_m',
                  'Certainty boosters': 'cert_rate',
                  'Hedge rate':         'hedge_rate'}

    fig, axes = plt.subplots(1, len(components), figsize=(15, 5))
    fig.suptitle('Production Cost Decomposition: What Are Believers Spending Effort On?\n'
                 '(Each component partial-regressed against alpha)',
                 fontsize=12)

    for ax, (name, col) in zip(axes, components.items()):
        for lbl in ('Yes','No'):
            sub = df_bin[df_bin['label']==lbl][[col,'alpha']].dropna()
            ax.scatter(sub[col], sub['alpha'], alpha=0.2, s=12,
                       color=C[lbl], edgecolors='none', label=lbl)
        ax.axhline(0, color='black', lw=0.8, ls=':')
        ax.set_xlabel(name); ax.set_ylabel('alpha')
        ax.set_title(f'alpha ~ {name}')

    axes[0].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT/'cost_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close(); print('  Saved: cost_decomposition.png')


def plot_alpha_canttell(df):
    """Where do 'Can't Tell' posts sit in alpha space?"""
    ct = df[df['label']=="Can't tell"]['alpha'].dropna()
    if len(ct) < 5:
        print("  Insufficient Can't Tell posts."); return

    yes_a = df[df['label']=='Yes']['alpha'].dropna()
    no_a  = df[df['label']=='No']['alpha'].dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    for vals, lbl in [(yes_a,'Believers (Yes)'), (no_a,"Non-Believers (No)"),
                      (ct,"Can't Tell")]:
        ax.hist(vals.clip(-8,8), bins=40, alpha=0.5, density=True,
                label=f'{lbl} (n={len(vals)})')
    ax.axvline(0, color='black', lw=1.2, ls='--', label='alpha=0 (indifferent speaker)')
    ax.set_xlabel('Rationality Parameter alpha')
    ax.set_ylabel('Density')
    ax.set_title("Alpha Distribution by Label Class\n"
                 "Higher alpha = speaker more 'optimised' to update listener's belief",
                 fontsize=11)
    ax.legend(fontsize=9)
    print(f"  Can't Tell alpha: median={ct.median():.3f}  "
          f"Yes: {yes_a.median():.3f}  No: {no_a.median():.3f}")
    plt.tight_layout()
    plt.savefig(OUT/'alpha_canttell.png', dpi=150, bbox_inches='tight')
    plt.close(); print('  Saved: alpha_canttell.png')


def plot_utility_scatter(df):
    """2D scatter: Cost vs Informativeness, coloured by alpha and label."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Utility Space: Production Cost vs. Informativeness\n'
                 'RSA Speaker optimises Utility = Informativeness - Cost',
                 fontsize=12)

    for ax, lbl in zip(axes, ['Yes','No']):
        sub = df[df['label']==lbl].copy()
        sc  = ax.scatter(sub['cost'], sub['inform'],
                         c=sub['alpha'].clip(-5,5),
                         cmap='RdYlGn', alpha=0.4, s=12, edgecolors='none')
        plt.colorbar(sc, ax=ax, label='alpha')
        ax.axhline(0, color='black', lw=0.8, ls=':')
        ax.axvline(sub['cost'].mean(), color='gray', lw=0.8, ls='--',
                   label='Mean cost')
        ax.set_xlabel('Production Cost C(u)'); ax.set_ylabel('Informativeness')
        ax.set_title(f'{"Believers" if lbl=="Yes" else "Non-Believers"} ({lbl})')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT/'utility_scatter.png', dpi=150, bbox_inches='tight')
    plt.close(); print('  Saved: utility_scatter.png')


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/train_rehydrated.jsonl')
    ap.add_argument('--rsa',  default='results/rsa_train_predictions.csv')
    ap.add_argument('--bn',   default='results/bn_predictions.csv')
    args = ap.parse_args()

    print(f'\n{"="*60}\nEXPERIMENT 2: PARAMETERIZED RSA — ALPHA ESTIMATION\n{"="*60}')

    df = load(args.data)
    n_ct = sum(df.label == "Can't tell")
    print(f'  Posts: {len(df)}  (Yes={sum(df.label=="Yes")}, '
          f'No={sum(df.label=="No")}, '
          f'CT={n_ct}')

    # Load posteriors if available
    loaded = False
    for fpath, col in [(args.rsa,'p_yes_rsa'), (args.bn,'p_yes_bn')]:
        if fpath and Path(fpath).exists():
            preds = pd.read_csv(fpath)
            if col in preds.columns:
                df = df.merge(preds[['id',col]].dropna(), on='id', how='left')
                df['p_posterior'] = df[col].fillna(0.5)
                print(f'  Using {col} as posterior from {fpath}')
                loaded = True; break
    if not loaded:
        print('  No pre-computed posteriors found — using feature-based proxy.')

    df = run_alpha_analysis(df)
    df.to_csv(RES/'alpha_estimates.csv', index=False)

    # Summary
    for lbl in ('Yes','No'):
        s = df[df['label']==lbl]['alpha'].dropna()
        print(f'  alpha {lbl}: mean={s.mean():.3f}  median={s.median():.3f}  '
              f'std={s.std():.3f}')

    _, p_mw = stats.mannwhitneyu(
        df[df['label']=='Yes']['alpha'].dropna(),
        df[df['label']=='No']['alpha'].dropna(),
        alternative='two-sided')
    print(f'  Mann-Whitney p-value: {p_mw:.4f}  '
          f'{"SIGNIFICANT" if p_mw<0.05 else "not significant"}')

    print('\nGenerating figures...')
    plot_alpha_distributions(df)
    plot_alpha_vs_density(df)
    plot_cost_decomposition(df)
    plot_alpha_canttell(df)
    plot_utility_scatter(df)

    print(f'\n{"="*50}\nSUMMARY')
    print('  Alpha = RSA rationality parameter per author.')
    print('  Higher alpha = speaker more optimised to update listener belief.')
    print(f'  Figures -> {OUT}/  |  Results -> {RES}/')

if __name__ == '__main__': main()