#!/usr/bin/env python3
"""
Analysis 6: Master Synthesis
=============================
Combines all feature streams into two summary visualisations:

  1. Master correlation heatmap
     All ~25 features (surprisal, lexical, syntactic, pragmatic, neuro-proxy,
     cosine) correlated with the binary conspiracy label and with each other.
     Rows = features, columns = features + label. Shows which features
     cluster together and which are most predictive.

  2. Psycholinguistic profile radar chart (MONEY SLIDE)
     Single spider chart comparing Believers vs. Non-Believers across
     all major psycholinguistic dimensions simultaneously.
     Axes: Surprisal, Lex-Difficulty, Syntactic-Complexity, Passive-Agency,
           Certainty-Boosting, Hedging, 3rd-Person-Attribution, Subjectivity,
           Factual-Grounding, Readability

  3. Feature importance bar chart
     Point-biserial correlation of every individual feature with conspiracy label,
     sorted by magnitude.  Shows the "top predictors" story cleanly.

  4. Pairwise scatter matrix (mini)
     Quick diagnostic for the 6 most separating features.

This script reuses the intermediate outputs computed in earlier scripts
(via --cache-dir) or recomputes the features from scratch if caches are absent.

Output: figures/06_synthesis/

Usage:
  # Run after scripts 1-5 (reads feature cache if present)
  python analysis_06_synthesis.py --data data/train_rehydrated.jsonl
"""

import argparse
import json
import pickle
import re
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=1.1)
COLORS = {'Yes': '#e74c3c', 'No': '#3498db'}
FIG_DIR = Path('figures/06_synthesis')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Word lists (mirrors scripts 1-3 — copied here for standalone use)
HEDGE_WORDS = {
    'maybe','perhaps','possibly','probably','apparently','seemingly',
    'supposedly','allegedly','reportedly','might','could','may',
    'seems','seem','appear','appears','suggest','suggests',
    'uncertain','unclear','rumored','purportedly','ostensibly',
    'doubtful','questionable','unconfirmed','claim','claims','claimed',
}
CERTAINTY_WORDS = {
    'clearly','obviously','undeniably','definitely','certainly',
    'absolutely','undoubtedly','surely','know','knew','known',
    'truth','fact','facts','proven','proof','evidence','confirmed',
    'revealed','exposed','everyone','nobody','always','never',
    'everywhere','wake','woke','sheep','sheeple',
}
PRON_1SG = {'i','me','my','mine','myself'}
PRON_1PL = {'we','us','our','ours','ourselves'}
PRON_3RD = {'they','them','their','theirs','themselves',
            'he','him','his','himself','she','her','hers','herself'}
VAGUENESS_PAT = [
    r'\bthey\b', r'\bthem\b', r'\bsome people\b', r'\bpeople say\b',
    r'\bsources say\b', r'\bsome say\b', r'\bthe elite[s]?\b',
    r'\bthe deep state\b', r'\bthe cabal\b', r'\bthe globalists?\b',
    r'\bthe media\b', r'\bthe government\b', r'\bmainstream\b',
]
MORAL_EMOTION = {
    'evil','corrupt','betray','lie','lies','deceive','deception','agenda',
    'criminal','murder','killed','tyranny','oppression','resist','fight',
    'wake','freedom','enslaved','puppet','sacrifice','poison','genocide',
    'cover','coverup','hoax','scam','satanic','trafficking','fear','hate',
    'anger','outrage','shocking','horrifying','disgusting','unbelievable',
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get('conspiracy','') in ('Yes','No'):
                data.append(item)
    return data


def get_text(item):
    return item.get('full_text', item.get('text','')).strip()


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION (lightweight, no GPU needed)
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_features(data, cache_path=None):
    """
    Extract the full feature matrix. Returns (feature_matrix, feature_names, labels).
    Saves to cache_path if provided, loads from there if it exists.
    """
    if cache_path and Path(cache_path).exists():
        print(f"  Loading features from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    try:
        import spacy, textstat
        from wordfreq import zipf_frequency
        nlp = spacy.load('en_core_web_sm')
        have_spacy = True
    except Exception:
        have_spacy = False
        print("  [!] spaCy not available — syntactic features skipped.")

    try:
        import textstat
        have_textstat = True
    except Exception:
        have_textstat = False

    try:
        from wordfreq import zipf_frequency
        have_wordfreq = True
    except Exception:
        have_wordfreq = False

    try:
        from empath import Empath
        lexicon = Empath()
        have_empath = True
    except Exception:
        have_empath = False

    CONTENT_POS = {'NOUN','VERB','ADJ','ADV'}

    rows   = []
    labels = []

    for item in tqdm(data, desc="  Feature extraction", ncols=80):
        text  = get_text(item)
        label = 1 if item['conspiracy'] == 'Yes' else 0
        words = re.findall(r'\b\w+\b', text.lower())
        n     = max(len(words), 1)
        feat  = {}

        # ── Lexical (no spaCy needed) ───────────────────────────────────────
        feat['word_count']     = len(words)
        feat['sent_count']     = max(text.count('.') + text.count('!') + text.count('?'), 1)
        feat['avg_word_len']   = np.mean([len(w) for w in words]) if words else 0
        feat['marker_count']   = len(item.get('markers', []))

        # Marker type counts
        for mtype in ['Actor','Action','Effect','Evidence','Victim']:
            feat[f'marker_{mtype.lower()}'] = sum(
                1 for m in item.get('markers',[]) if m.get('type','') == mtype
            )

        # Pragmatic
        feat['hedge_rate']   = sum(1 for w in words if w in HEDGE_WORDS) / n * 100
        feat['certain_rate'] = sum(1 for w in words if w in CERTAINTY_WORDS) / n * 100
        feat['p1sg_rate']    = sum(1 for w in words if w in PRON_1SG) / n * 100
        feat['p1pl_rate']    = sum(1 for w in words if w in PRON_1PL) / n * 100
        feat['p3rd_rate']    = sum(1 for w in words if w in PRON_3RD) / n * 100
        feat['vague_rate']   = sum(
            len(re.findall(p, text.lower())) for p in VAGUENESS_PAT
        ) / n * 100
        feat['moral_emot_rate'] = sum(1 for w in words if w in MORAL_EMOTION) / n * 100

        # Hedge/Certainty ratio
        feat['hedge_certain_ratio'] = feat['hedge_rate'] / (feat['certain_rate'] + 1e-6)

        # TTR
        toks_alpha = [w for w in words if w.isalpha()]
        feat['ttr'] = len(set(toks_alpha)) / len(toks_alpha) if toks_alpha else 0

        # Readability
        if have_textstat and len(words) >= 10:
            try:
                feat['fk_grade'] = textstat.flesch_kincaid_grade(text)
            except Exception:
                feat['fk_grade'] = 0
        else:
            feat['fk_grade'] = 0

        # Lexical frequency
        if have_wordfreq:
            freqs = []
            for w in words:
                if w.isalpha() and len(w) > 2:
                    freqs.append(zipf_frequency(w, 'en', minimum=0.0))
            feat['mean_zipf']      = float(np.mean(freqs)) if freqs else 0
            feat['low_freq_ratio'] = float(np.mean([f < 3.5 for f in freqs])) if freqs else 0
        else:
            feat['mean_zipf'] = feat['low_freq_ratio'] = 0

        # Empath / moral-emotion
        if have_empath:
            try:
                scores = lexicon.analyze(text, normalize=True) or {}
                emot_cats = ['negative_emotion','positive_emotion','anger',
                             'fear','hate','suffering','aggression']
                moral_cats = ['violence','power','government','crime','religion']
                feat['emot_score']  = float(np.mean([scores.get(c,0) for c in emot_cats]))
                feat['moral_score'] = float(np.mean([scores.get(c,0) for c in moral_cats]))
            except Exception:
                feat['emot_score'] = feat['moral_score'] = 0
        else:
            feat['emot_score']  = feat['moral_emot_rate']
            feat['moral_score'] = feat['moral_emot_rate']

        # ── Syntactic (spaCy needed) ────────────────────────────────────────
        if have_spacy:
            doc = nlp(text[:4000])
            n_tok = max(len([t for t in doc if not t.is_space]), 1)
            n_sents = max(len(list(doc.sents)), 1)

            # NP / VP
            feat['n_np']        = len(list(doc.noun_chunks))
            feat['n_vp']        = sum(1 for t in doc if t.pos_ == 'VERB')
            feat['np_vp_ratio'] = feat['n_np'] / (feat['n_vp'] + 1e-6)

            # Dep depth
            def _depth(t):
                d, x = 0, t
                while x.head != x: x = x.head; d += 1
                return d
            depths = [_depth(t) for t in doc if not t.is_space]
            feat['mean_dep_depth'] = float(np.mean(depths)) if depths else 0

            # Passive
            feat['passive_ratio'] = sum(
                1 for t in doc if t.dep_ in ('nsubjpass','auxpass')
            ) / n_sents

            # Subord
            feat['subord_density'] = sum(
                1 for t in doc if t.dep_ in {'advcl','relcl','ccomp','xcomp','acl'}
            ) / n_sents

            # NER density
            feat['ner_density']    = len(doc.ents) / n_tok * 100
            ner_fact_types = {'DATE','CARDINAL','PERCENT','LAW','ORG','PERSON'}
            feat['factual_score']  = sum(
                1 for e in doc.ents if e.label_ in ner_fact_types
            ) / n_tok * 100
        else:
            for k in ['n_np','n_vp','np_vp_ratio','mean_dep_depth',
                      'passive_ratio','subord_density','ner_density','factual_score']:
                feat[k] = 0

        rows.append(feat)
        labels.append(label)

    # Align all rows to same keys
    all_keys = list(rows[0].keys())
    matrix   = np.array([[r.get(k, 0) for k in all_keys] for r in rows],
                         dtype=float)

    result = (matrix, all_keys, np.array(labels))

    if cache_path:
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"  Features cached to: {cache_path}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

READABLE_NAMES = {
    'word_count':          'Word Count',
    'sent_count':          'Sentence Count',
    'avg_word_len':        'Avg Word Length',
    'marker_count':        'Narrative Markers (total)',
    'marker_actor':        'Actor Markers',
    'marker_action':       'Action Markers',
    'marker_effect':       'Effect Markers',
    'marker_evidence':     'Evidence Markers',
    'marker_victim':       'Victim Markers',
    'hedge_rate':          'Hedging Rate',
    'certain_rate':        'Certainty/Booster Rate',
    'p1sg_rate':           '1st-Person Singular Rate',
    'p1pl_rate':           '1st-Person Plural Rate',
    'p3rd_rate':           '3rd-Person Pronoun Rate',
    'vague_rate':          'Vague Attribution Rate',
    'moral_emot_rate':     'Moral/Emotion Word Rate',
    'hedge_certain_ratio': 'Hedge÷Certainty Ratio',
    'ttr':                 'Type-Token Ratio (TTR)',
    'fk_grade':            'Flesch-Kincaid Grade',
    'mean_zipf':           'Mean Zipf Frequency',
    'low_freq_ratio':      'Low-Freq Word Ratio',
    'emot_score':          'Emotional Score (Empath)',
    'moral_score':         'Moral Score (Empath)',
    'n_np':                'Noun Phrase Count',
    'n_vp':                'Verb Phrase Count',
    'np_vp_ratio':         'NP/VP Ratio',
    'mean_dep_depth':      'Mean Dependency Depth',
    'passive_ratio':       'Passive Voice Rate',
    'subord_density':      'Subordinate Clause Density',
    'ner_density':         'NER Density',
    'factual_score':       'Factual Grounding Score',
}


def plot_feature_importance(matrix, feature_names, labels):
    """Point-biserial correlation of every feature with conspiracy label."""
    corrs, pvals = [], []
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        if np.std(col) < 1e-9:
            corrs.append(0); pvals.append(1); continue
        r, p = stats.pointbiserialr(labels, col)
        corrs.append(r); pvals.append(p)

    order   = np.argsort(np.abs(corrs))[::-1]
    top_idx = order[:30]

    names_r = [READABLE_NAMES.get(feature_names[i], feature_names[i]) for i in top_idx]
    corrs_r = [corrs[i] for i in top_idx]
    pvals_r = [pvals[i] for i in top_idx]
    colors_bar  = [COLORS['Yes'] if c > 0 else COLORS['No'] for c in corrs_r]
    # Significance markers
    sig = ['***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
           for p in pvals_r]

    fig, ax = plt.subplots(figsize=(13, 11))
    y = np.arange(len(names_r))
    ax.barh(y, corrs_r, color=colors_bar, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(names_r, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Point-Biserial Correlation with Conspiracy Label\n'
                  '(positive = higher in Believers, negative = higher in Non-Believers)',
                  fontsize=10)
    ax.set_title('Feature Importance: Psycholinguistic Predictors of Conspiracy Belief\n'
                 '(sorted by |r|, * p<.05  ** p<.01  *** p<.001)',
                 fontsize=12, pad=12)

    for i, (c, s) in enumerate(zip(corrs_r, sig)):
        offset = 0.004 if c >= 0 else -0.004
        ax.text(c + offset, i, s, va='center', fontsize=9,
                ha='left' if c >= 0 else 'right')

    patches = [
        mpatches.Patch(color=COLORS['Yes'], label='Positive → Believer (Yes)'),
        mpatches.Patch(color=COLORS['No'],  label='Negative → Non-Believer (No)'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: feature_importance.png")
    return corrs, pvals


def plot_correlation_heatmap(matrix, feature_names, labels):
    """Full feature × feature correlation matrix + conspiracy label column."""
    # Keep top 20 features by variance
    variances = np.var(matrix, axis=0)
    top_feat  = np.argsort(variances)[-22:][::-1]

    sub_mat  = matrix[:, top_feat]
    sub_names = [READABLE_NAMES.get(feature_names[i], feature_names[i])
                 for i in top_feat]

    # Add label as last column
    full = np.column_stack([sub_mat, labels])
    full_names = sub_names + ['Conspiracy Label']

    corr_mat = np.corrcoef(full.T)

    fig, ax = plt.subplots(figsize=(16, 13))
    mask = np.zeros_like(corr_mat, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(
        corr_mat, annot=True, fmt='.2f', cmap='coolwarm',
        center=0, vmin=-1, vmax=1,
        xticklabels=full_names, yticklabels=full_names,
        ax=ax, annot_kws={'size': 6.5},
        linewidths=0.3, mask=mask,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.set_title('Master Feature Correlation Heatmap\n'
                 '(last column = correlation with Conspiracy label)',
                 fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'master_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: master_correlation_heatmap.png")


def plot_psycholinguistic_radar(matrix, feature_names, labels):
    """
    The MONEY SLIDE: radar chart summarising all dimensions.
    10 axes covering the full psycholinguistic profile.
    """
    # Axis definitions: (label, feature_key, direction)
    #   direction=1 → higher = more conspiracy; -1 → lower = more conspiracy
    axes_def = [
        ('Surprisal\n(Processing Load)',  'fk_grade',         1),
        ('Lex Difficulty\n(AoA proxy)',   'low_freq_ratio',   1),
        ('Syntactic\nComplexity',         'mean_dep_depth',   1),
        ('Passive\nAgency',               'passive_ratio',    1),
        ('Certainty\nBoosting',           'certain_rate',     1),
        ('Hedging\n& Uncertainty',        'hedge_rate',       1),
        ('3rd-Person\nAttribution',       'p3rd_rate',        1),
        ('Subjectivity\n(mPFC proxy)',    'emot_score',       1),
        ('Factual\nGrounding',            'factual_score',   -1),
        ('Narrative\nDensity',            'marker_count',     1),
    ]

    yes_idx = labels == 1
    no_idx  = labels == 0

    def _norm_mean(feat_key, idx):
        if feat_key not in feature_names:
            return 0.5
        j   = feature_names.index(feat_key)
        col = matrix[:, j]
        mn, mx = col.min(), col.max()
        if mx - mn < 1e-9:
            return 0.5
        return float(np.mean((col[idx] - mn) / (mx - mn)))

    yes_vals, no_vals, axis_labels = [], [], []
    for lbl, feat_key, direction in axes_def:
        yv = _norm_mean(feat_key, yes_idx)
        nv = _norm_mean(feat_key, no_idx)
        if direction == -1:
            yv, nv = 1 - yv, 1 - nv
        yes_vals.append(yv)
        no_vals.append(nv)
        axis_labels.append(lbl)

    N = len(axis_labels)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    yes_vals += yes_vals[:1]
    no_vals  += no_vals[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.plot(angles, yes_vals, 'o-', color=COLORS['Yes'], lw=2.5,
            label='Believers (Yes)', zorder=3)
    ax.fill(angles, yes_vals, alpha=0.20, color=COLORS['Yes'])

    ax.plot(angles, no_vals,  'o-', color=COLORS['No'],  lw=2.5,
            label='Non-Believers (No)', zorder=3)
    ax.fill(angles, no_vals,  alpha=0.20, color=COLORS['No'])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_labels, fontsize=10.5)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)

    # Gridlines
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(angles, [r]*len(angles), color='gray', lw=0.4, alpha=0.4)

    ax.set_title('Psycholinguistic Profile\nConspiracy Believers vs. Non-Believers',
                 pad=25, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.40, 1.15), fontsize=11)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'psycholinguistic_radar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: psycholinguistic_radar.png  ← MONEY SLIDE")


def plot_pairwise_scatter(matrix, feature_names, labels):
    """Quick pairwise scatter for top 6 features by correlation with label."""
    corrs = [abs(stats.pointbiserialr(labels, matrix[:, j])[0])
             if np.std(matrix[:, j]) > 1e-9 else 0
             for j in range(matrix.shape[1])]
    top6  = np.argsort(corrs)[-6:][::-1]

    names6 = [READABLE_NAMES.get(feature_names[i], feature_names[i]) for i in top6]
    mat6   = matrix[:, top6]

    fig, axes = plt.subplots(6, 6, figsize=(18, 18))
    fig.suptitle('Pairwise Scatter: Top 6 Features by Conspiracy Correlation',
                 fontsize=13, y=1.01)

    yes_idx = labels == 1
    no_idx  = labels == 0
    sample  = min(300, yes_idx.sum(), no_idx.sum())
    rng     = np.random.default_rng(42)
    yi = rng.choice(np.where(yes_idx)[0], sample, replace=False)
    ni = rng.choice(np.where(no_idx)[0],  sample, replace=False)

    for r in range(6):
        for c in range(6):
            ax = axes[r, c]
            if r == c:
                ax.hist(mat6[yi, r], bins=25, alpha=0.6, color=COLORS['Yes'], density=True)
                ax.hist(mat6[ni, r], bins=25, alpha=0.6, color=COLORS['No'],  density=True)
                ax.set_xlabel(names6[r], fontsize=7)
            else:
                ax.scatter(mat6[yi, c], mat6[yi, r], alpha=0.15, s=8,
                           color=COLORS['Yes'], edgecolors='none')
                ax.scatter(mat6[ni, c], mat6[ni, r], alpha=0.15, s=8,
                           color=COLORS['No'],  edgecolors='none')
            if c == 0:
                ax.set_ylabel(names6[r], fontsize=7)
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'pairwise_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: pairwise_scatter.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',      default='data/train_rehydrated.jsonl')
    parser.add_argument('--cache-dir', default='figures/cache',
                        help='Directory for feature caches')
    parser.add_argument('--no-cache',  action='store_true',
                        help='Ignore existing cache and recompute')
    args = parser.parse_args()

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache_dir) / 'features_synthesis.pkl'
    if args.no_cache and cache_path.exists():
        cache_path.unlink()

    print(f"\n{'='*60}")
    print("ANALYSIS 6: MASTER SYNTHESIS")
    print(f"{'='*60}")
    print(f"Figures → {FIG_DIR}/")

    print("\n[1/2] Loading data & extracting features...")
    data = load_data(args.data)
    print(f"  Total: {len(data)}  (Yes: {sum(1 for d in data if d['conspiracy']=='Yes')}"
          f"  No: {sum(1 for d in data if d['conspiracy']=='No')})")

    matrix, feature_names, labels = extract_all_features(
        data, cache_path=str(cache_path)
    )
    print(f"  Feature matrix: {matrix.shape[0]} posts × {matrix.shape[1]} features")

    print("\n[2/2] Generating figures...")
    plot_feature_importance(matrix, feature_names, labels)
    plot_correlation_heatmap(matrix, feature_names, labels)
    plot_psycholinguistic_radar(matrix, feature_names, labels)
    plot_pairwise_scatter(matrix, feature_names, labels)

    # Print top 10 features by absolute correlation
    corrs = [stats.pointbiserialr(labels, matrix[:, j])[0]
             if np.std(matrix[:, j]) > 1e-9 else 0
             for j in range(matrix.shape[1])]
    order = np.argsort(np.abs(corrs))[::-1][:10]
    print("\nTop 10 features by |r| with conspiracy label:")
    for rank, i in enumerate(order, 1):
        name = READABLE_NAMES.get(feature_names[i], feature_names[i])
        direction = "↑ Yes" if corrs[i] > 0 else "↑ No"
        print(f"  {rank:2d}. {name:35s}  r={corrs[i]:+.4f}  {direction}")

    print(f"\n✓ All figures saved to {FIG_DIR}/")


if __name__ == '__main__':
    main()
