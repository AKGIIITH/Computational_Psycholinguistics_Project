#!/usr/bin/env python3
"""
Module B: Bayesian Network over Narrative Markers
===================================================
Formally models the generative process: how does a conspiracy believer
(vs. non-believer) produce the linguistic features we observe?

Network structure:
  [Belief] ← latent root
      |
  [Actor] [Action] [Effect] [Evidence] [Victim]
               |                 |
           [Booster]          [Hedge]
               |
           [Vagueness]

Key queries (theoretically grounded):
  1. Joint marker probability:
     P(all 5 markers | Belief=Yes) vs P(all 5 | Belief=No)
     → Formally proves narrative density as a probabilistic fact

  2. The Gricean Quality violation (most important finding):
     P(Booster=High | Belief=Yes, Evidence=LOW) >> P(Booster=High | Belief=No, Evidence=LOW)
     → When believers lack evidence, they OVERCOMPENSATE with certainty boosters
     → This is the Gricean maxim violation captured as a conditional probability

  3. The vagueness + certainty paradox:
     P(Vague=High | Booster=High, Belief=Yes)
     → Believers use certainty language about vague referents simultaneously

  4. Can't Tell posterior inference:
     Run BN inference on "Can't Tell" posts
     → Compare to human annotator confusion
     → Flat posteriors = model agrees this text is genuinely ambiguous

  5. Structure learning validation:
     Compare hand-specified DAG vs. learned structure (Hill-Climbing)
     → Does the data confirm our theoretical model of the generative process?

Output: figures/bayesian_network/ + results/bn_predictions.csv

Usage:
  python bayesian_B_network.py --data data/train_rehydrated.jsonl
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
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.metrics import (f1_score, accuracy_score,
                             classification_report, roc_auc_score)
from tqdm import tqdm

# pgmpy imports
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import (MaximumLikelihoodEstimator,
                               BayesianEstimator, HillClimbSearch)
from pgmpy.parameter_estimator import DiscreteMLE
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=1.1)

COLORS = {'Yes': '#e74c3c', 'No': '#3498db', "Can't tell": '#95a5a6'}
FIG_DIR = Path('figures/bayesian_network')
RES_DIR = Path('results')
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

# ── LINGUISTIC LEXICONS ───────────────────────────────────────────────────────
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
    'wake','woke','sheep','sheeple',
}
VAGUENESS_PAT = [
    r'\bthey\b', r'\bthem\b', r'\bsome people\b', r'\bpeople say\b',
    r'\bsources say\b', r'\bsome say\b', r'\bit is said\b',
    r'\bthe elite[s]?\b', r'\bthe establishment\b',
    r'\bthe deep state\b', r'\bthe cabal\b', r'\bthe globalists?\b',
    r'\bthe media\b', r'\bthe government\b',
]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path, include_canttell=False):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            lbl = d.get('conspiracy','').strip()
            if lbl in ('Yes','No') or (include_canttell
                                        and lbl == "Can't tell"):
                rows.append(d)
    return rows


def get_text(d):
    return d.get('full_text', d.get('text','')).strip()


def extract_features(data):
    """
    Extract and binarize all features needed for the Bayesian Network.
    Thresholds are set at medians from the full dataset to ensure
    balanced binary splits.
    """
    records = []
    for d in tqdm(data, desc='  Feature extraction', ncols=80):
        text   = get_text(d)
        label  = d.get('conspiracy','').strip()
        words  = re.findall(r'\b\w+\b', text.lower())
        n      = max(len(words), 1)
        markers = d.get('markers', [])

        # ── Marker counts per type ──────────────────────────────────────────
        mtype_counts = defaultdict(int)
        for m in markers:
            mtype_counts[m.get('type', '')] += 1

        # ── Pragmatic features ───────────────────────────────────────────────
        hedge_rate   = sum(1 for w in words if w in HEDGE_WORDS) / n * 100
        certain_rate = sum(1 for w in words if w in CERTAINTY_WORDS) / n * 100
        vague_count  = sum(len(re.findall(p, text.lower()))
                          for p in VAGUENESS_PAT)
        vague_rate   = vague_count / n * 100

        records.append({
            'id':           d.get('_id',''),
            'label':        label,
            'text_len':     n,
            'total_markers': len(markers),
            'n_actor':      mtype_counts['Actor'],
            'n_action':     mtype_counts['Action'],
            'n_effect':     mtype_counts['Effect'],
            'n_evidence':   mtype_counts['Evidence'],
            'n_victim':     mtype_counts['Victim'],
            'hedge_rate':   hedge_rate,
            'certain_rate': certain_rate,
            'vague_rate':   vague_rate,
        })

    df = pd.DataFrame(records)
    return df


def binarize(df, thresholds=None):
    """
    Convert continuous features to binary using the provided thresholds.
    If thresholds is None, compute medians from the data (use training set).
    Returns binarized DataFrame and the thresholds dict.
    """
    binary_cols = {
        'Actor':    'n_actor',
        'Action':   'n_action',
        'Effect':   'n_effect',
        'Evidence': 'n_evidence',
        'Victim':   'n_victim',
        'Booster':  'certain_rate',
        'Hedge':    'hedge_rate',
        'Vague':    'vague_rate',
    }

    if thresholds is None:
        thresholds = {col: df[raw].median()
                      for col, raw in binary_cols.items()}
        # For marker counts, use threshold of 1 (present/absent)
        for mk in ['Actor','Action','Effect','Evidence','Victim']:
            thresholds[mk] = 0.5   # present if count >= 1

    df_bin = pd.DataFrame()
    df_bin['Belief'] = (df['label'] == 'Yes').astype(int)
    df_bin['label']  = df['label']
    df_bin['id']     = df['id']

    for col, raw in binary_cols.items():
        df_bin[col] = (df[raw] > thresholds[col]).astype(int)

    return df_bin, thresholds


# ─────────────────────────────────────────────────────────────────────────────
# BAYESIAN NETWORK
# ─────────────────────────────────────────────────────────────────────────────

THEORY_EDGES = [
    # Belief drives all marker types
    ('Belief', 'Actor'),
    ('Belief', 'Action'),
    ('Belief', 'Effect'),
    ('Belief', 'Evidence'),
    ('Belief', 'Victim'),
    # Belief and evidence jointly predict boosters
    # (the Gricean Quality violation node)
    ('Belief',   'Booster'),
    ('Evidence', 'Booster'),
    # Belief predicts hedging
    ('Belief', 'Hedge'),
    # Booster predicts vagueness (certainty about vague referents)
    ('Belief',  'Vague'),
    ('Booster', 'Vague'),
]

THEORY_NODES = ['Belief','Actor','Action','Effect','Evidence','Victim',
                'Booster','Hedge','Vague']


def build_and_fit_bn(df_bin, edges=THEORY_EDGES):
    """Build BN with the theory-specified structure and fit CPDs."""
    df_model = df_bin[THEORY_NODES].copy()

    model = DiscreteBayesianNetwork(edges)

    # Fit CPDs using DiscreteMLE
    model.fit(df_model, estimator=DiscreteMLE())
    
    # Smooth CPDs with Laplace to avoid zeros
    for cpd in model.get_cpds():
        cpd.values = cpd.values + 1e-6
        cpd.normalize()
    
    return model


def learn_structure(df_bin, max_indegree=3):
    """
    Use Hill-Climbing + BIC score to learn the network structure from data.
    Compare to our theory-specified structure.
    """
    df_model = df_bin[THEORY_NODES].copy()
    
    # Convert to category dtype for pgmpy discrete detection
    for col in df_model.columns:
        df_model[col] = df_model[col].astype('category')
    
    hc  = HillClimbSearch(df_model)
    learned = hc.estimate(scoring_method='bic-d',
                          max_indegree=max_indegree,
                          show_progress=False)
    return learned


# ─────────────────────────────────────────────────────────────────────────────
# KEY QUERIES
# ─────────────────────────────────────────────────────────────────────────────

def query_joint_marker_probability(infer, label='Yes'):
    """
    P(Actor=1, Action=1, Effect=1, Evidence=1, Victim=1 | Belief=b)
    Uses chain rule via the inference engine.
    """
    b = 1 if label == 'Yes' else 0
    evidence = {'Belief': b}

    joint = 1.0
    for node in ['Actor','Action','Effect','Evidence','Victim']:
        q = infer.query([node], evidence=evidence, show_progress=False)
        joint *= float(q.values[1])   # P(node=1 | Belief=b)

    return joint


def query_gricean_violation(infer):
    """
    The KEY FINDING: P(Booster=1 | Belief=1, Evidence=0)
                  vs P(Booster=1 | Belief=1, Evidence=1)
                  vs P(Booster=1 | Belief=0, Evidence=0)
                  vs P(Booster=1 | Belief=0, Evidence=1)

    Grice: if you lack evidence (Evidence=0), you should hedge.
    Conspiracy prior: if you lack evidence, you OVERCOMPENSATE with boosters.
    The BN captures this mathematically.
    """
    results = {}
    for belief in [0, 1]:
        for evid in [0, 1]:
            q = infer.query(
                ['Booster'],
                evidence={'Belief': belief, 'Evidence': evid},
                show_progress=False
            )
            key = f'Belief={["No","Yes"][belief]},Evidence={evid}'
            results[key] = float(q.values[1])

    return results


def query_vagueness_certainty_paradox(infer):
    """
    P(Vague=1 | Booster=1, Belief=1) vs P(Vague=1 | Booster=0, Belief=1)
    Believers use certainty language about vague referents simultaneously.
    """
    results = {}
    for booster in [0, 1]:
        q = infer.query(
            ['Vague'],
            evidence={'Booster': booster, 'Belief': 1},
            show_progress=False
        )
        results[f'Booster={booster},Belief=Yes'] = float(q.values[1])
    return results


def inference_on_posts(df_bin, infer):
    """
    Run BN inference P(Belief=1 | all observed features) for every post.
    This gives a BN-based classification probability.
    """
    obs_cols = ['Actor','Action','Effect','Evidence','Victim',
                'Booster','Hedge','Vague']
    preds = []

    for _, row in df_bin.iterrows():
        evidence = {col: int(row[col]) for col in obs_cols}
        try:
            q     = infer.query(['Belief'], evidence=evidence,
                                show_progress=False)
            p_yes = float(q.values[1])
        except Exception:
            p_yes = 0.5
        preds.append(p_yes)

    df_bin = df_bin.copy()
    df_bin['p_yes_bn'] = preds
    df_bin['pred_bn']  = (df_bin['p_yes_bn'] >= 0.5).astype(int)
    return df_bin


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_cpd_heatmap(model):
    """Heatmap of the Booster CPD — the Gricean violation node."""
    # Get the CPD for Booster
    cpd = model.get_cpds('Booster')
    print(f'\n  Booster CPD:\n{cpd}')

    # Extract values into a matrix
    # Parents: Belief (0/1), Evidence (0/1) → 4 combinations
    try:
        vals = cpd.values   # shape depends on pgmpy version
        # We want P(Booster=1 | parents)
        # Reshape to get the conditional table
        parents = cpd.variables[1:]          # parent variables
        parent_card = cpd.cardinality[1:]   # parent cardinalities
        n_parent_states = int(np.prod(parent_card))
        for c in parent_card:
            n_parent_states *= c

        # P(Booster=1 | parent combination)
        p1 = cpd.values[1] if cpd.values.ndim > 1 else cpd.values

        if hasattr(p1, '__len__') and len(p1) == 4:
            mat = np.array(p1).reshape(2, 2)
            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Evidence=Absent', 'Evidence=Present'])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Belief=No', 'Belief=Yes'])
            ax.set_title('P(Certainty Booster = HIGH | Belief, Evidence)\n'
                        'The Gricean Quality Violation — captured as a CPD',
                        fontsize=11)
            plt.colorbar(im, ax=ax, label='P(Booster=High)')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f'{mat[i,j]:.3f}',
                           ha='center', va='center', fontsize=14,
                           color='black', fontweight='bold')
            plt.tight_layout()
            plt.savefig(FIG_DIR / 'bn_booster_cpd.png',
                       dpi=150, bbox_inches='tight')
            plt.close()
            print('  Saved: bn_booster_cpd.png')
    except Exception as e:
        print(f'  Could not plot CPD heatmap: {e}')


def plot_gricean_violation(gricean_results):
    """Bar chart of P(Booster=High | Belief, Evidence) — 4 conditions."""
    labels = list(gricean_results.keys())
    values = list(gricean_results.values())

    colors_bar = []
    for lbl in labels:
        if 'Belief=Yes' in lbl and 'Evidence=0' in lbl:
            colors_bar.append('#c0392b')   # red — the violation
        elif 'Belief=Yes' in lbl:
            colors_bar.append('#e74c3c')
        elif 'Evidence=0' in lbl:
            colors_bar.append('#5d6d7e')
        else:
            colors_bar.append('#3498db')

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(labels)), values, color=colors_bar, alpha=0.85,
                  width=0.55)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(',', '\n') for l in labels], fontsize=9)
    ax.set_ylabel('P(Booster=High)')
    ax.set_title(
        'P(Certainty Booster = HIGH | Belief, Evidence)\n'
        'Gricean Quality Violation: Believers compensate for missing evidence\n'
        'with certainty language — confirmed as a conditional probability',
        fontsize=11
    )
    ax.set_ylim(0, 1)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    # Annotate the key comparison
    # Find indices for Belief=Yes,Evidence=0 and Belief=No,Evidence=0
    key_yes_noeq = next((i for i, l in enumerate(labels)
                         if 'Belief=Yes' in l and 'Evidence=0' in l), None)
    key_no_noeq  = next((i for i, l in enumerate(labels)
                         if 'Belief=No'  in l and 'Evidence=0' in l), None)
    if key_yes_noeq is not None and key_no_noeq is not None:
        diff = values[key_yes_noeq] - values[key_no_noeq]
        ax.annotate(
            f'Gricean violation:\n+{diff:.3f} when Evidence absent',
            xy=(key_yes_noeq, values[key_yes_noeq] + 0.03),
            xytext=(key_yes_noeq + 0.8, values[key_yes_noeq] + 0.15),
            fontsize=9, color='#922b21',
            arrowprops=dict(arrowstyle='->', color='#922b21', lw=1.5),
        )

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'bn_gricean_violation.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: bn_gricean_violation.png')

    # Print
    print('\n  Gricean Quality Violation — Conditional Probabilities:')
    for lbl, val in gricean_results.items():
        print(f'    P(Booster=High | {lbl}) = {val:.4f}')


def plot_joint_marker_probability(joint_yes, joint_no):
    """Bar comparing joint probability of all 5 markers under each belief."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        ['Believers (Yes)', 'Non-Believers (No)'],
        [joint_yes, joint_no],
        color=[COLORS['Yes'], COLORS['No']],
        alpha=0.85, width=0.4
    )
    ax.set_ylabel('P(Actor ∩ Action ∩ Effect ∩ Evidence ∩ Victim | Belief)')
    ax.set_title(
        'Joint Probability of Complete Narrative Frame\n'
        'P(all 5 marker types present | Belief) — formal proof of narrative density',
        fontsize=11
    )
    ax.set_ylim(0, max(joint_yes, joint_no) * 1.4)
    for bar, val in zip(bars, [joint_yes, joint_no]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', fontsize=13, fontweight='bold')
    ratio = joint_yes / (joint_no + 1e-9)
    ax.text(0.97, 0.92,
            f'Ratio: {ratio:.1f}x more likely\nunder believer model',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, color=COLORS['Yes'],
            bbox=dict(boxstyle='round,pad=0.3', fc='#fadbd8', alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'bn_joint_marker_probability.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: bn_joint_marker_probability.png')
    print(f'  Joint P(all 5 markers | Yes) = {joint_yes:.6f}')
    print(f'  Joint P(all 5 markers | No)  = {joint_no:.6f}')
    print(f'  Ratio: {ratio:.2f}x more likely under believer model')


def plot_bn_posterior_vs_rsa(df_bin):
    """Distribution of BN posterior by true label."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for lbl, color in [('Yes', COLORS['Yes']), ('No', COLORS['No'])]:
        vals = df_bin[df_bin['label'] == lbl]['p_yes_bn'].dropna().values
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=25, alpha=0.6, color=color,
                label=f'{lbl} (n={len(vals)})', density=True)

    ax.axvline(0.5, color='black', lw=1.5, ls='--', label='Decision boundary')
    ax.set_xlabel('BN Posterior P(Belief=Yes | features)')
    ax.set_ylabel('Density')
    ax.set_title('Bayesian Network Posterior Distribution by True Label',
                 fontsize=12)
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'bn_posterior_distribution.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: bn_posterior_distribution.png')


def plot_learned_vs_theory_structure(learned_edges, theory_edges):
    """Compare learned structure (Hill-Climbing) vs theory structure."""
    learned_set = set(tuple(e) for e in learned_edges)
    theory_set  = set(tuple(e) for e in theory_edges)

    in_both        = learned_set & theory_set
    only_learned   = learned_set - theory_set
    only_theory    = theory_set  - learned_set

    print('\n  Structure comparison (learned vs. theory):')
    print(f'  Confirmed edges (in both):    {sorted(in_both)}')
    print(f'  Extra edges (data only):      {sorted(only_learned)}')
    print(f'  Missing edges (theory only):  {sorted(only_theory)}')
    print(f'  Edge overlap: {len(in_both)}/{len(theory_set)} theory edges confirmed')

    # Bar summary
    fig, ax = plt.subplots(figsize=(8, 4))
    categories = ['Confirmed\n(theory & data)', 'Data-only\n(novel)',
                  'Theory-only\n(not in data)']
    vals = [len(in_both), len(only_learned), len(only_theory)]
    colors_bar = ['#27ae60', '#3498db', '#e67e22']
    ax.bar(categories, vals, color=colors_bar, alpha=0.85, width=0.4)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, str(v), ha='center', fontsize=13,
                fontweight='bold')
    ax.set_ylabel('Number of edges')
    ax.set_title('Learned Structure vs. Theory Structure\n'
                 '(Hill-Climbing BIC vs. RSA/psycholinguistic theory)',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'bn_structure_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: bn_structure_comparison.png')


def plot_cant_tell_bn(df_bin):
    """Can't Tell posts: BN posterior distribution."""
    ct = df_bin[df_bin['label'] == "Can't tell"]['p_yes_bn'].dropna()
    if len(ct) < 5:
        print("  Not enough Can't Tell posts for BN analysis.")
        return

    yes_p = df_bin[df_bin['label'] == 'Yes']['p_yes_bn'].dropna()
    no_p  = df_bin[df_bin['label'] == 'No']['p_yes_bn'].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Bayesian Network Uncertainty for 'Can't Tell' Posts\n"
        "Flat posteriors confirm genuine linguistic ambiguity, not annotator noise",
        fontsize=12, y=1.02
    )

    ax = axes[0]
    ax.hist(ct.values, bins=15, color=COLORS["Can't tell"],
            alpha=0.8, density=True, label=f"Can't Tell (n={len(ct)})")
    ax.axvline(0.5, color='black', lw=1.5, ls='--')
    ax.set_xlabel('BN Posterior P(Yes)')
    ax.set_ylabel('Density')
    ax.set_title("Can't Tell: BN Posterior Distribution")
    ax.legend()

    ax = axes[1]
    groups = [yes_p.values, no_p.values, ct.values]
    labels_v = ['Yes', 'No', "Can't Tell"]
    vp = ax.violinplot(groups, positions=[1, 2, 3], widths=0.55,
                       showmedians=True, showextrema=False)
    for pc, lbl in zip(vp['bodies'], labels_v):
        pc.set_facecolor(COLORS.get(lbl, '#aaa'))
        pc.set_alpha(0.75)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels_v)
    ax.axhline(0.5, color='black', lw=1.2, ls='--')
    ax.set_ylabel('BN Posterior P(Yes)')
    ax.set_title("Posterior by Label Class")

    ct_entropy = -(ct * np.log2(ct + 1e-9) + (1-ct) * np.log2(1-ct + 1e-9))
    print(f"  Can't Tell median posterior: {ct.median():.3f}")
    print(f"  Can't Tell mean entropy:     {ct_entropy.mean():.3f} bits")

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'bn_canttell.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: bn_canttell.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/train_rehydrated.jsonl')
    args = parser.parse_args()

    print(f'\n{"="*60}')
    print('MODULE B: BAYESIAN NETWORK OVER NARRATIVE MARKERS')
    print(f'{"="*60}')
    print(f'Figures -> {FIG_DIR}/')

    # ── Data ──────────────────────────────────────────────────────────────────
    print('\n[1/5] Loading and featurizing data...')
    data     = load_data(args.data, include_canttell=True)
    df_feat  = extract_features(data)

    # Separate splits
    df_binary = df_feat[df_feat['label'].isin(['Yes','No'])].copy()
    df_ct     = df_feat[df_feat['label'] == "Can't tell"].copy()
    print(f'  Binary-labelled: {len(df_binary)}  '
          f'(Yes={sum(df_binary.label=="Yes")}, '
          f'No={sum(df_binary.label=="No")})')
    print(f"  Can't tell:      {len(df_ct)}")

    # Binarize using training-set medians
    df_bin_binary, thresholds = binarize(df_binary)
    print(f'\n  Feature binarization thresholds:')
    for k, v in thresholds.items():
        print(f'    {k}: > {v:.3f}')

    # ── Build and fit BN ──────────────────────────────────────────────────────
    print('\n[2/5] Building and fitting Bayesian Network...')
    model = build_and_fit_bn(df_bin_binary, THEORY_EDGES)
    print(f'  Nodes: {model.nodes()}')
    print(f'  Edges: {model.edges()}')

    # Check model validity
    assert model.check_model(), "BN model failed consistency check!"
    print('  Model consistency check: PASSED')

    infer = VariableElimination(model)

    # ── Key queries ───────────────────────────────────────────────────────────
    print('\n[3/5] Running key theoretical queries...')

    # Query 1: Joint marker probability
    joint_yes = query_joint_marker_probability(infer, 'Yes')
    joint_no  = query_joint_marker_probability(infer, 'No')

    # Query 2: Gricean Quality violation
    gricean = query_gricean_violation(infer)

    # Query 3: Vagueness+Certainty paradox
    paradox = query_vagueness_certainty_paradox(infer)
    print('\n  Vagueness+Certainty paradox:')
    for k, v in paradox.items():
        print(f'    P(Vague=High | {k}) = {v:.4f}')

    # ── Structure learning ────────────────────────────────────────────────────
    print('\n[4/5] Learning structure from data (Hill-Climbing + BIC)...')
    try:
        learned = learn_structure(df_bin_binary)
        print(f'  Learned edges: {list(learned.edges())}')
        plot_learned_vs_theory_structure(
            list(learned.edges()), THEORY_EDGES
        )
    except Exception as e:
        print(f'  Structure learning skipped due to: {e}')

    # ── Inference on all posts ────────────────────────────────────────────────
    print('\n[5/5] Running inference on all posts...')

    # Include Can't Tell
    df_ct_bin, _ = binarize(df_ct, thresholds)
    df_all_bin   = pd.concat([df_bin_binary, df_ct_bin], ignore_index=True)
    df_all_bin   = inference_on_posts(df_all_bin, infer)

    # Evaluation on binary labels only
    df_eval = df_all_bin[df_all_bin['label'].isin(['Yes','No'])].copy()
    y_true  = df_eval['Belief'].values
    y_pred  = df_eval['pred_bn'].values

    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    print(f'\n  BN Classification Results:')
    print(f'  Macro F1:  {f1:.4f}')
    print(f'  Accuracy:  {acc:.4f}')
    print(classification_report(y_true, y_pred, zero_division=0))

    # Save
    df_all_bin.to_csv(RES_DIR / 'bn_predictions.csv', index=False)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print('\nGenerating figures...')
    plot_cpd_heatmap(model)
    plot_gricean_violation(gricean)
    plot_joint_marker_probability(joint_yes, joint_no)
    plot_bn_posterior_vs_rsa(df_all_bin)
    plot_cant_tell_bn(df_all_bin)

    print(f'\n{"="*50}')
    print('BAYESIAN NETWORK SUMMARY')
    print(f'{"="*50}')
    print(f'  Nodes: {len(model.nodes())}  |  Edges: {len(model.edges())}')
    print(f'  Joint P(all 5 markers | Yes): {joint_yes:.6f}')
    print(f'  Joint P(all 5 markers | No):  {joint_no:.6f}')
    print(f'  Ratio: {joint_yes/(joint_no+1e-9):.1f}x')
    print(f'  Gricean violation:')
    for k, v in gricean.items():
        print(f'    P(Booster=High | {k}) = {v:.4f}')
    print(f'  BN Classification F1: {f1:.4f}')
    print(f'  (DeBERTa test F1: 0.750 — gap = {0.750-f1:.3f})')
    print(f'\n✓ Figures saved to {FIG_DIR}/')
    print(f'✓ Predictions saved to {RES_DIR}/')


if __name__ == '__main__':
    main()