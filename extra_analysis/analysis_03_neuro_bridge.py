#!/usr/bin/env python3
"""
Analysis 3: Neuroscience-to-NLP Bridge
=======================================
Translates the double dissociation from the Nature paper into measurable
NLP features, creating a direct linguistic mirror of brain activation.

Brain → Language proxy mapping:
  Hippocampus / Precuneus (Non-believers, fact-checking):
      → NER density: named entities, dates, numbers, specific referents
      → "Factual Grounding Score"

  vmPFC / dmPFC (Believers, value-based belief processing):
      → Empath emotional/moral language density
      → "Subjective Salience Score"

Analyses:
  1. NER density per class (entities per 100 tokens, broken down by type)
  2. Empath category scores (emotional, moral, cognitive)
  3. 2-axis scatter: every post on Factual vs Subjective axes (the killer slide)
  4. Heatmap of all neuro-proxy features correlated to conspiracy label
  5. Marker type overlap: do higher NER posts also have more Evidence markers?

Output: figures/03_neuro_bridge/

Usage:
  python analysis_03_neuro_bridge.py --data data/train_rehydrated.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import spacy

try:
    from empath import Empath
    EMPATH_AVAILABLE = True
except ImportError:
    EMPATH_AVAILABLE = False
    print("[WARNING] empath not installed → install with: pip install empath")
    print("          Subjective Salience proxy will use a manual word list instead.")

sns.set_theme(style='whitegrid', font_scale=1.1)
COLORS = {'Yes': '#e74c3c', 'No': '#3498db'}
FIG_DIR = Path('figures/03_neuro_bridge')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Empath categories of interest (mPFC proxy)
EMPATH_CATS = [
    'negative_emotion', 'positive_emotion', 'anger', 'fear', 'trust',
    'violence', 'power', 'government', 'crime', 'death', 'religion',
    'social_media', 'hate', 'suffering', 'aggression', 'gain', 'loss',
    'confusion', 'shame', 'dispute',
]

# Fallback moral/emotional word list if Empath not available
MORAL_EMOTION_WORDS = {
    'evil', 'corrupt', 'betray', 'lie', 'lies', 'deceive', 'deception',
    'agenda', 'criminal', 'murder', 'killed', 'tyranny', 'oppression',
    'resist', 'fight', 'wake', 'truth', 'freedom', 'enslaved', 'puppet',
    'sacrifice', 'poison', 'genocide', 'cover', 'coverup', 'hoax', 'scam',
    'evil', 'satanic', 'pedophile', 'trafficking', 'ritual', 'fear', 'hate',
    'anger', 'outrage', 'shocking', 'horrifying', 'disgusting', 'unbelievable',
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get('conspiracy', '') in ('Yes', 'No'):
                data.append(item)
    return data


def get_text(item):
    return item.get('full_text', item.get('text', '')).strip()


def get_markers_by_type(item):
    counts = {'Actor': 0, 'Action': 0, 'Effect': 0, 'Evidence': 0, 'Victim': 0}
    for m in item.get('markers', []):
        t = m.get('type', '')
        if t in counts:
            counts[t] += 1
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# NER DENSITY (Hippocampus proxy)
# ─────────────────────────────────────────────────────────────────────────────

NER_ENTITY_TYPES = ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'TIME',
                    'MONEY', 'PERCENT', 'CARDINAL', 'LAW', 'EVENT', 'NORP']


def compute_ner_density(texts, nlp):
    results = []
    for text in tqdm(texts, desc="  NER density", ncols=80):
        if not text:
            results.append(None)
            continue
        doc = nlp(text[:5000])
        n_tokens = max(len([t for t in doc if not t.is_space]), 1)

        total_ents = len(doc.ents)
        by_type = {et: 0 for et in NER_ENTITY_TYPES}
        for ent in doc.ents:
            if ent.label_ in by_type:
                by_type[ent.label_] += 1

        # Factual grounding score: emphasise DATE, CARDINAL, PERCENT, LAW, ORG
        factual_score = (
            by_type['DATE'] + by_type['CARDINAL'] + by_type['PERCENT'] +
            by_type['LAW']  + by_type['ORG']      + by_type['PERSON']
        ) / n_tokens * 100

        results.append({
            'total_ent_density': total_ents / n_tokens * 100,
            'factual_score':     factual_score,
            'by_type':           by_type,
            'n_tokens':          n_tokens,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# EMPATH / MORAL-EMOTIONAL DENSITY (mPFC proxy)
# ─────────────────────────────────────────────────────────────────────────────

def compute_empath_scores(texts):
    results = []

    if EMPATH_AVAILABLE:
        lexicon = Empath()

    for text in tqdm(texts, desc="  Empath (mPFC proxy)", ncols=80):
        if not text:
            results.append(None)
            continue

        n_words = max(len(text.split()), 1)

        if EMPATH_AVAILABLE:
            try:
                scores = lexicon.analyze(text, categories=EMPATH_CATS,
                                         normalize=True)
                if scores is None:
                    scores = {c: 0.0 for c in EMPATH_CATS}
            except Exception:
                scores = {c: 0.0 for c in EMPATH_CATS}

            # mPFC proxy: average of emotional / moral categories
            emot_cats = ['negative_emotion', 'positive_emotion',
                         'anger', 'fear', 'hate', 'suffering',
                         'aggression', 'shame']
            moral_cats = ['violence', 'power', 'government',
                          'crime', 'religion', 'dispute']

            emot_score  = np.mean([scores.get(c, 0) for c in emot_cats])
            moral_score = np.mean([scores.get(c, 0) for c in moral_cats])
            subj_score  = (emot_score + moral_score) / 2.0

            results.append({
                'subjective_score': float(subj_score),
                'emot_score':       float(emot_score),
                'moral_score':      float(moral_score),
                'empath_detail':    {k: float(v) for k, v in scores.items()},
            })
        else:
            # Fallback: simple moral/emotional word count
            words = text.lower().split()
            moral_count = sum(1 for w in words if w in MORAL_EMOTION_WORDS)
            subj_score  = moral_count / n_words * 100
            results.append({
                'subjective_score': subj_score,
                'emot_score':       subj_score,
                'moral_score':      subj_score,
                'empath_detail':    {},
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_ner_breakdown(yes_ner, no_ner):
    types_show = ['PERSON', 'ORG', 'GPE', 'DATE', 'CARDINAL', 'LAW']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Named Entity Density (Hippocampus Proxy: Fact Retrieval)',
                 fontsize=13, y=1.02)

    x = np.arange(len(types_show))
    w = 0.35

    for ax, ner_list, color, label in [
        (axes[0], yes_ner, COLORS['Yes'], 'Believers (Yes)'),
        (axes[1], no_ner,  COLORS['No'],  'Non-Believers (No)'),
    ]:
        means, sems = [], []
        for et in types_show:
            vals = [r['by_type'].get(et, 0) / r['n_tokens'] * 100
                    for r in ner_list if r]
            means.append(np.mean(vals) if vals else 0)
            sems.append(stats.sem(vals) if len(vals) > 1 else 0)
        ax.bar(x, means, width=0.6, color=color, alpha=0.8,
               yerr=sems, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(types_show, rotation=30, ha='right')
        ax.set_ylabel('Entities per 100 tokens')
        ax.set_title(label)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'ner_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: ner_breakdown.png")


def plot_factual_vs_subjective(yes_ner, no_ner, yes_emp, no_emp):
    """THE KILLER SLIDE: scatter plot mapping the brain's double dissociation."""
    fig, ax = plt.subplots(figsize=(11, 9))

    for ner_list, emp_list, color, label in [
        (yes_ner, yes_emp, COLORS['Yes'], 'Believers (Yes)'),
        (no_ner,  no_emp,  COLORS['No'],  'Non-Believers (No)'),
    ]:
        factual = [n['factual_score']     for n, e in zip(ner_list, emp_list)
                   if n and e]
        subj    = [e['subjective_score']  for n, e in zip(ner_list, emp_list)
                   if n and e]
        if not factual:
            continue

        ax.scatter(factual, subj, alpha=0.35, color=color, s=30,
                   edgecolors='none', label=label)

        # Plot class centroid
        cx, cy = np.mean(factual), np.mean(subj)
        ax.scatter([cx], [cy], color=color, s=250, edgecolors='black',
                   linewidths=1.5, zorder=5)
        ax.annotate(f'{label}\ncentroid', (cx, cy),
                    textcoords='offset points', xytext=(8, 6),
                    fontsize=9, color=color, fontweight='bold')

    # Quadrant labels
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xm, ym = (xlim[0]+xlim[1])/2, (ylim[0]+ylim[1])/2
    ax.axvline(xm, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(ym, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.text(xlim[0]+0.01, ylim[1]-0.001, '← Low Factual / High Subjective\n  (mPFC dominant)',
            fontsize=8, color='gray', va='top')
    ax.text(xlim[1]-0.01, ylim[1]-0.001, 'High Factual / High Subjective →',
            fontsize=8, color='gray', va='top', ha='right')
    ax.text(xlim[0]+0.01, ylim[0]+0.001,
            '← Low Factual / Low Subjective\n  (sparse text)',
            fontsize=8, color='gray', va='bottom')
    ax.text(xlim[1]-0.01, ylim[0]+0.001,
            'High Factual / Low Subjective →\n  (Hippocampus dominant)',
            fontsize=8, color='gray', va='bottom', ha='right')

    ax.set_xlabel('Factual Grounding Score\n(NER density: dates, names, orgs, laws)',
                  fontsize=11)
    ax.set_ylabel('Subjective Salience Score\n(Empath: emotional + moral language density)',
                  fontsize=11)
    ax.set_title(
        'Neuroscience-to-NLP Bridge: The Double Dissociation\n'
        'Nature paper (2025): Believers → vmPFC/dmPFC (value-based)\n'
        'Non-Believers → Hippocampus/Precuneus (fact retrieval)',
        fontsize=12, pad=12
    )
    ax.legend(markerscale=2, fontsize=10)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'factual_vs_subjective_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: factual_vs_subjective_scatter.png  ← THE KILLER SLIDE")


def plot_neuro_feature_heatmap(yes_ner, no_ner, yes_emp, no_emp,
                                yes_data, no_data):
    """Heatmap: mean feature value per class for all neuro-proxy features."""
    feature_names = (
        ['NER total density', 'Factual score', 'PERSON dens.', 'ORG dens.',
         'GPE dens.', 'DATE dens.', 'LAW dens.', 'CARDINAL dens.'] +
        ['Subjective score', 'Emot. score', 'Moral score'] +
        ['Evidence markers', 'Actor markers', 'Victim markers']
    )

    def _safe_mean(lst):
        c = [v for v in lst if v is not None and np.isfinite(v)]
        return np.mean(c) if c else 0.0

    yes_row, no_row = [], []

    # NER features
    for key in ['total_ent_density', 'factual_score']:
        yes_row.append(_safe_mean([r[key] for r in yes_ner if r]))
        no_row.append( _safe_mean([r[key] for r in no_ner  if r]))

    for et in ['PERSON', 'ORG', 'GPE', 'DATE', 'LAW', 'CARDINAL']:
        yes_row.append(_safe_mean([r['by_type'].get(et, 0)/r['n_tokens']*100
                                   for r in yes_ner if r]))
        no_row.append( _safe_mean([r['by_type'].get(et, 0)/r['n_tokens']*100
                                   for r in no_ner  if r]))

    # Empath features
    for key in ['subjective_score', 'emot_score', 'moral_score']:
        yes_row.append(_safe_mean([e[key] for e in yes_emp if e]))
        no_row.append( _safe_mean([e[key] for e in no_emp  if e]))

    # Marker type features
    for mtype in ['Evidence', 'Actor', 'Victim']:
        yes_row.append(_safe_mean([get_markers_by_type(d)[mtype]
                                   for d in yes_data]))
        no_row.append( _safe_mean([get_markers_by_type(d)[mtype]
                                   for d in no_data]))

    # Build matrix and normalise row-wise for visual comparison
    matrix = np.array([yes_row, no_row])
    row_max = matrix.max(axis=0, keepdims=True) + 1e-9
    matrix_norm = matrix / row_max

    fig, ax = plt.subplots(figsize=(max(14, len(feature_names)*0.8), 4))
    im = ax.imshow(matrix_norm, aspect='auto', cmap='RdBu_r',
                   vmin=0, vmax=1)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=40, ha='right', fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Believers\n(Yes)', 'Non-Believers\n(No)'])
    ax.set_title('Neuro-Proxy Feature Heatmap\n'
                 '(colour = normalised mean value per class)',
                 fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04, label='Normalised score')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'neuro_feature_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: neuro_feature_heatmap.png")


def plot_empath_breakdown(yes_emp, no_emp):
    """Bar chart of top Empath categories for each class."""
    if not EMPATH_AVAILABLE:
        print("  [SKIP] Empath breakdown (empath not installed)")
        return

    cats = EMPATH_CATS
    yes_means = []
    no_means  = []
    for c in cats:
        yv = [e['empath_detail'].get(c, 0) for e in yes_emp if e and e['empath_detail']]
        nv = [e['empath_detail'].get(c, 0) for e in no_emp  if e and e['empath_detail']]
        yes_means.append(np.mean(yv) if yv else 0)
        no_means.append( np.mean(nv) if nv else 0)

    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - 0.2, yes_means, 0.38, label='Believers (Yes)',
           color=COLORS['Yes'], alpha=0.8)
    ax.bar(x + 0.2, no_means,  0.38, label='Non-Believers (No)',
           color=COLORS['No'],  alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Empath score (normalised)')
    ax.set_title('Empath Category Breakdown\n'
                 '(mPFC proxy: emotional & moral language activation)',
                 fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'empath_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: empath_breakdown.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/train_rehydrated.jsonl')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("ANALYSIS 3: NEUROSCIENCE-TO-NLP BRIDGE")
    print(f"{'='*60}")
    if not EMPATH_AVAILABLE:
        print("[!] empath not installed. Using fallback word list.")
        print("    Install: pip install empath")
    print(f"Figures → {FIG_DIR}/")

    data = load_data(args.data)
    yes_data = [d for d in data if d['conspiracy'] == 'Yes']
    no_data  = [d for d in data if d['conspiracy'] == 'No']
    print(f"  Yes: {len(yes_data)}  |  No: {len(no_data)}")

    yes_texts = [get_text(d) for d in yes_data]
    no_texts  = [get_text(d) for d in no_data]

    print("\n[1/3] Loading spaCy (with NER)...")
    nlp = spacy.load('en_core_web_sm')

    print("\n[2/3] NER density (Hippocampus proxy)...")
    yes_ner = compute_ner_density(yes_texts, nlp)
    no_ner  = compute_ner_density(no_texts,  nlp)

    print("\n[3/3] Empath scores (mPFC proxy)...")
    yes_emp = compute_empath_scores(yes_texts)
    no_emp  = compute_empath_scores(no_texts)

    # Stats summary
    for label, ner_list, emp_list in [
        ('Yes (Believer)',    yes_ner, yes_emp),
        ('No (Non-Believer)', no_ner,  no_emp),
    ]:
        fs = np.mean([r['factual_score']     for r in ner_list if r])
        ss = np.mean([e['subjective_score']  for e in emp_list if e])
        print(f"  {label}: factual={fs:.4f}  subjective={ss:.4f}")

    print("\nGenerating figures...")
    plot_ner_breakdown(yes_ner, no_ner)
    plot_factual_vs_subjective(yes_ner, no_ner, yes_emp, no_emp)
    plot_neuro_feature_heatmap(yes_ner, no_ner, yes_emp, no_emp, yes_data, no_data)
    plot_empath_breakdown(yes_emp, no_emp)

    print(f"\n✓ All figures saved to {FIG_DIR}/")


if __name__ == '__main__':
    main()
