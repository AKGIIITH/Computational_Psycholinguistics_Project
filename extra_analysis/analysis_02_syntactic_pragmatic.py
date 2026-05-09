#!/usr/bin/env python3
"""
Analysis 2: Syntactic Profiling & Epistemic Stance (Pragmatics)
================================================================
Syntactic and pragmatic analyses connecting linguistic form to
cognitive processing load and speaker stance.

Analyses:
  1.  NP / VP counts and ratio (nominal vs verbal style)
  2.  Mean dependency tree depth (Gibson DLT: depth ∝ integration cost)
  3.  Passive voice frequency (hidden-agency constructions)
  4.  Subordinate clause density (syntactic embedding)
  5.  SVO triplet extraction (Subject-Verb-Object narrative motifs)
  6.  NetworkX network graph of top SVO triplets per class
  7.  Hedging word density (might, could, seems, possibly…)
  8.  Certainty/booster density (clearly, undeniably, everyone knows…)
  9.  Pronoun profile: 1st-singular / 1st-plural / 3rd-person ratios
  10. Attribution-vagueness markers ("they", "some say", "sources say"…)
  11. Epistemic-stance radar chart summarising axes 7-10

Theoretical grounding:
  - DLT (Gibson 1998): deeper syntactic dependencies → longer reading time
  - Gricean Maxim of Quality: booster/hedge asymmetry signals unverifiable claims
  - Conspiracy discourse: passive voice hides agency; 3rd-person "they" = out-group

Output: figures/02_syntactic/ (PNG files)

Usage:
  python analysis_02_syntactic_pragmatic.py --data data/train_rehydrated.jsonl
"""

import argparse
import json
import re
from collections import Counter, defaultdict
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
import networkx as nx

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
COLORS = {'Yes': '#e74c3c', 'No': '#3498db'}
FIG_DIR = Path('figures/02_syntactic')
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# LEXICONS
# ─────────────────────────────────────────────────────────────────────────────

HEDGE_WORDS = {
    'maybe', 'perhaps', 'possibly', 'probably', 'apparently', 'seemingly',
    'supposedly', 'allegedly', 'reportedly', 'might', 'could', 'may',
    'seems', 'seem', 'appear', 'appears', 'suggest', 'suggests',
    'uncertain', 'unclear', 'rumored', 'rumoured', 'purportedly',
    'ostensibly', 'doubtful', 'questionable', 'unconfirmed', 'claim',
    'claims', 'claimed', 'assert', 'asserts', 'asserted',
}

CERTAINTY_WORDS = {
    'clearly', 'obviously', 'undeniably', 'definitely', 'certainly',
    'absolutely', 'undoubtedly', 'unquestionably', 'obviously', 'plainly',
    'surely', 'know', 'knew', 'known', 'truth', 'fact', 'facts',
    'proven', 'proof', 'evidence', 'confirmed', 'revealed', 'exposed',
    'everyone', 'nobody', 'always', 'never', 'everywhere',
    'wake', 'woke', 'sheep', 'sheeple',
}

VAGUENESS_PHRASES = [
    r'\bthey\b', r'\bthem\b', r'\bthose people\b', r'\bsome people\b',
    r'\bpeople say\b', r'\bsources say\b', r'\bsome say\b', r'\bit is said\b',
    r'\bthe elite[s]?\b', r'\bthe establishment\b', r'\bthe powers that be\b',
    r'\bthe deep state\b', r'\bthe cabal\b', r'\bthe globalists?\b',
    r'\bthe media\b', r'\bthe government\b', r'\bmainstream\b',
]

PRON_1SG  = {'i', 'me', 'my', 'mine', 'myself'}
PRON_1PL  = {'we', 'us', 'our', 'ours', 'ourselves'}
PRON_3RD  = {'they', 'them', 'their', 'theirs', 'themselves',
             'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself'}


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


# ─────────────────────────────────────────────────────────────────────────────
# SYNTACTIC FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def dep_depth(token):
    d = 0
    t = token
    while t.head != t:
        t = t.head
        d += 1
    return d


def extract_syntactic(texts, nlp):
    results = []
    for text in tqdm(texts, desc="  Syntactic parsing", ncols=80):
        if not text:
            results.append(None)
            continue
        doc = nlp(text[:5000])
        words = [t for t in doc if not t.is_space]
        if not words:
            results.append(None)
            continue

        n_tokens = len(words)

        # NP / VP counts
        nps = list(doc.noun_chunks)
        n_np = len(nps)
        vps = [t for t in doc if t.pos_ == 'VERB']
        n_vp = len(vps)
        np_vp_ratio = n_np / (n_vp + 1e-6)

        # Dependency depth (mean across all tokens)
        depths = [dep_depth(t) for t in words]
        mean_dep_depth = float(np.mean(depths))
        max_dep_depth  = float(max(depths))

        # Passive voice: look for auxpass or nsubjpass dependency
        n_passive = sum(
            1 for t in doc
            if t.dep_ in ('nsubjpass', 'auxpass') or
               (t.dep_ == 'aux' and t.lemma_ in ('be', 'get') and
                t.head.tag_ in ('VBN', 'VBD'))
        )
        n_sents = max(1, len(list(doc.sents)))
        passive_ratio = n_passive / n_sents

        # Subordinate clause density: SBAR-like (mark, advcl, relcl, ccomp, xcomp)
        subord_deps = {'advcl', 'relcl', 'ccomp', 'xcomp', 'acl', 'mark'}
        n_subord = sum(1 for t in doc if t.dep_ in subord_deps)
        subord_density = n_subord / n_sents

        # SVO extraction
        svos = []
        for sent in doc.sents:
            for tok in sent:
                if tok.dep_ == 'nsubj':
                    verb = tok.head
                    obj = next(
                        (ch for ch in verb.children if ch.dep_ in ('dobj', 'attr', 'pobj')),
                        None
                    )
                    if obj:
                        svos.append((
                            tok.lemma_.lower(),
                            verb.lemma_.lower(),
                            obj.lemma_.lower()
                        ))

        results.append({
            'n_np':           n_np,
            'n_vp':           n_vp,
            'np_vp_ratio':    np_vp_ratio,
            'mean_dep_depth': mean_dep_depth,
            'max_dep_depth':  max_dep_depth,
            'passive_ratio':  passive_ratio,
            'subord_density': subord_density,
            'svos':           svos,
            'n_tokens':       n_tokens,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PRAGMATIC / EPISTEMIC FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_pragmatic(texts):
    results = []
    for text in tqdm(texts, desc="  Pragmatic markers", ncols=80):
        if not text:
            results.append(None)
            continue
        words = re.findall(r'\b\w+\b', text.lower())
        n = max(len(words), 1)

        hedge    = sum(1 for w in words if w in HEDGE_WORDS)
        certain  = sum(1 for w in words if w in CERTAINTY_WORDS)
        p1sg     = sum(1 for w in words if w in PRON_1SG)
        p1pl     = sum(1 for w in words if w in PRON_1PL)
        p3rd     = sum(1 for w in words if w in PRON_3RD)

        vague = sum(
            len(re.findall(p, text.lower()))
            for p in VAGUENESS_PHRASES
        )

        results.append({
            'hedge_rate':    hedge   / n * 100,
            'certain_rate':  certain / n * 100,
            'p1sg_rate':     p1sg    / n * 100,
            'p1pl_rate':     p1pl    / n * 100,
            'p3rd_rate':     p3rd    / n * 100,
            'vague_rate':    vague   / n * 100,
            'hedge_certain_ratio': hedge / (certain + 1e-6),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def _violin(ax, yes_vals, no_vals, ylabel, title):
    yc = [v for v in yes_vals if v is not None and np.isfinite(v)]
    nc = [v for v in no_vals  if v is not None and np.isfinite(v)]
    if not yc or not nc:
        ax.set_title(f'{title}\n(no data)')
        return
    vp = ax.violinplot([yc, nc], positions=[1, 2], widths=0.6,
                       showmedians=True, showextrema=False)
    for pc, col in zip(vp['bodies'], [COLORS['Yes'], COLORS['No']]):
        pc.set_facecolor(col)
        pc.set_alpha(0.75)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Believer\n(Yes)', 'Non-Believer\n(No)'])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(yc) > 1 and len(nc) > 1:
        _, p = stats.mannwhitneyu(yc, nc, alternative='two-sided')
        star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(0.97, 0.96, f'p={p:.4f} {star}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))


def plot_syntactic(yes_syn, no_syn):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Syntactic Profiling: Conspiracy vs. Non-Conspiracy',
                 fontsize=14, y=1.01)

    _violin(axes[0,0],
            [s['np_vp_ratio']    for s in yes_syn if s],
            [s['np_vp_ratio']    for s in no_syn  if s],
            'NP/VP Ratio', 'NP / VP Ratio\n(higher = more nominal)')

    _violin(axes[0,1],
            [s['mean_dep_depth'] for s in yes_syn if s],
            [s['mean_dep_depth'] for s in no_syn  if s],
            'Mean Depth', 'Mean Dependency Depth\n(Gibson DLT: depth ∝ reading cost)')

    _violin(axes[0,2],
            [s['passive_ratio']  for s in yes_syn if s],
            [s['passive_ratio']  for s in no_syn  if s],
            'Passives / Sentence', 'Passive Voice Rate\n("we are being lied to…")')

    _violin(axes[1,0],
            [s['subord_density'] for s in yes_syn if s],
            [s['subord_density'] for s in no_syn  if s],
            'Subord. Clauses / Sent.', 'Subordinate Clause Density\n(syntactic complexity)')

    _violin(axes[1,1],
            [s['n_np']           for s in yes_syn if s],
            [s['n_np']           for s in no_syn  if s],
            'Count', 'Noun Phrase Count')

    _violin(axes[1,2],
            [s['n_vp']           for s in yes_syn if s],
            [s['n_vp']           for s in no_syn  if s],
            'Count', 'Verb Phrase Count')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'syntactic_profile.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: syntactic_profile.png")

    # Print key stats
    for key, label in [('np_vp_ratio', 'NP/VP ratio'),
                        ('mean_dep_depth', 'Dep depth'),
                        ('passive_ratio', 'Passive/sent')]:
        yv = [s[key] for s in yes_syn if s]
        nv = [s[key] for s in no_syn  if s]
        _, p = stats.mannwhitneyu(yv, nv, alternative='two-sided')
        print(f"    {label}: Yes={np.mean(yv):.3f}  No={np.mean(nv):.3f}  p={p:.4f}")


def plot_svo_networks(yes_syn, no_syn, top_n=20):
    """Build and draw SVO network graphs for each class."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle('SVO Narrative Motif Networks\n'
                 '(nodes = subjects/objects, edges = verbs, weight = frequency)',
                 fontsize=13)

    for ax, syn_list, color, title in [
        (axes[0], yes_syn, COLORS['Yes'], 'Conspiracy Believers (Yes)'),
        (axes[1], no_syn,  COLORS['No'],  'Non-Believers (No)'),
    ]:
        all_svos = []
        for s in syn_list:
            if s:
                all_svos.extend(s['svos'])

        counter = Counter(all_svos)
        top = counter.most_common(top_n)

        G = nx.DiGraph()
        for (subj, verb, obj), cnt in top:
            if subj and obj and verb:
                edge_label = verb
                if G.has_edge(subj, obj):
                    G[subj][obj]['weight'] += cnt
                    G[subj][obj]['label'] += f'\n{verb}'
                else:
                    G.add_edge(subj, obj, weight=cnt, label=verb)

        if not G.nodes():
            ax.text(0.5, 0.5, 'No SVO data', ha='center', va='center')
            ax.set_title(title)
            continue

        pos = nx.spring_layout(G, seed=42, k=1.5)
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_w   = max(weights) if weights else 1

        nx.draw_networkx_nodes(G, pos, ax=ax,
                               node_size=400, node_color=color, alpha=0.7)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
        nx.draw_networkx_edges(G, pos, ax=ax,
                               width=[2*w/max_w + 0.5 for w in weights],
                               edge_color='#555', arrows=True,
                               arrowsize=10, connectionstyle='arc3,rad=0.1')
        edge_labels = nx.get_edge_attributes(G, 'label')
        # Trim long labels
        edge_labels = {k: v.split('\n')[0][:10] for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                     font_size=6, label_pos=0.35)
        ax.set_title(title, fontsize=12, pad=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'svo_networks.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: svo_networks.png")


def plot_pragmatic(yes_prag, no_prag):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Epistemic Stance & Pragmatic Markers\n'
                 '(Gricean Quality violations: boosters compensate for unverifiable claims)',
                 fontsize=13, y=1.02)

    pairs = [
        ('hedge_rate',   'Hedges per 100 words',      'Hedging Rate\n(might, could, possibly…)',         axes[0,0]),
        ('certain_rate', 'Certainty words / 100 words', 'Certainty / Booster Rate\n(clearly, truth, fact…)', axes[0,1]),
        ('p3rd_rate',    '3rd-Person Pronouns / 100',  '3rd-Person Pronoun Rate\n(they/them = the conspirators)', axes[0,2]),
        ('p1pl_rate',    '1st-Plural Pronouns / 100',  '1st-Plural Pronoun Rate\n(we/us = in-group identity)',   axes[1,0]),
        ('p1sg_rate',    '1st-Singular Pronouns / 100', '1st-Singular Pronoun Rate\n(I/me = personal stance)',   axes[1,1]),
        ('vague_rate',   'Vague Attribution / 100',    'Attribution Vagueness\n("some say", "the elites"…)',     axes[1,2]),
    ]

    for key, ylabel, title, ax in pairs:
        _violin(ax,
                [p[key] for p in yes_prag if p],
                [p[key] for p in no_prag  if p],
                ylabel, title)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'pragmatic_markers.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: pragmatic_markers.png")


def plot_epistemic_radar(yes_prag, no_prag):
    """Spider/radar chart for epistemic stance dimensions."""
    keys   = ['hedge_rate', 'certain_rate', 'p1sg_rate', 'p1pl_rate',
              'p3rd_rate', 'vague_rate']
    labels = ['Hedging', 'Certainty', '1st-Singular\n(I/me)',
              '1st-Plural\n(we/us)', '3rd-Person\n(they)', 'Vagueness']

    def _mean(prag, key):
        vals = [p[key] for p in prag if p]
        return np.mean(vals) if vals else 0.0

    yes_vals = [_mean(yes_prag, k) for k in keys]
    no_vals  = [_mean(no_prag,  k) for k in keys]

    # Normalize to 0-1 range for visual clarity
    max_vals = [max(y, n) + 1e-9 for y, n in zip(yes_vals, no_vals)]
    yes_norm = [v / m for v, m in zip(yes_vals, max_vals)]
    no_norm  = [v / m for v, m in zip(no_vals,  max_vals)]

    N = len(keys)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    yes_norm += yes_norm[:1]
    no_norm  += no_norm[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, yes_norm, 'o-', color=COLORS['Yes'], lw=2.5,
            label='Believers (Yes)')
    ax.fill(angles, yes_norm, alpha=0.2, color=COLORS['Yes'])

    ax.plot(angles, no_norm,  'o-', color=COLORS['No'],  lw=2.5,
            label='Non-Believers (No)')
    ax.fill(angles, no_norm,  alpha=0.2, color=COLORS['No'])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels([])
    ax.set_title('Epistemic Stance Profile\n'
                 '(values normalised to max per dimension)',
                 pad=20, fontsize=13)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'epistemic_radar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: epistemic_radar.png")

    # Print raw values
    print("\n  Epistemic Stance Summary:")
    for k, lbl in zip(keys, labels):
        yv = _mean(yes_prag, k)
        nv = _mean(no_prag,  k)
        print(f"    {lbl.split(chr(10))[0]:20s}: Yes={yv:.4f}  No={nv:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/train_rehydrated.jsonl')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("ANALYSIS 2: SYNTACTIC PROFILING & PRAGMATIC MARKERS")
    print(f"{'='*60}")
    print(f"Figures → {FIG_DIR}/")

    data = load_data(args.data)
    yes_data = [d for d in data if d['conspiracy'] == 'Yes']
    no_data  = [d for d in data if d['conspiracy'] == 'No']
    print(f"  Yes: {len(yes_data)}  |  No: {len(no_data)}")

    yes_texts = [get_text(d) for d in yes_data]
    no_texts  = [get_text(d) for d in no_data]

    print("\n[1/3] Loading spaCy (en_core_web_sm)...")
    # Need parser for dependency parse
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])

    print("\n[2/3] Syntactic features...")
    yes_syn = extract_syntactic(yes_texts, nlp)
    no_syn  = extract_syntactic(no_texts,  nlp)

    print("\n[3/3] Pragmatic / epistemic features...")
    yes_prag = compute_pragmatic(yes_texts)
    no_prag  = compute_pragmatic(no_texts)

    print("\nGenerating figures...")
    plot_syntactic(yes_syn, no_syn)
    plot_svo_networks(yes_syn, no_syn)
    plot_pragmatic(yes_prag, no_prag)
    plot_epistemic_radar(yes_prag, no_prag)

    print(f"\n✓ All figures saved to {FIG_DIR}/")


if __name__ == '__main__':
    main()
