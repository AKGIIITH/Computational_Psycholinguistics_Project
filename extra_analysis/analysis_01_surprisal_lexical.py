#!/usr/bin/env python3
"""
Analysis 1: Surprisal & Lexical Processing Load
================================================
Psycholinguistic analyses grounded in information-theoretic models.

Analyses performed:
  1.  GPT-2 per-word surprisal  (mean, variance, positional trajectory)
  2.  Surprisal at clause boundaries vs. mid-clause
  3.  Lexical frequency via wordfreq (Zipf scores for content words)
  4.  Low-frequency word ratio  (words below Zipf 3.5)
  5.  Readability: Flesch-Kincaid Grade Level & SMOG Index
  6.  Lexical diversity: Type-Token Ratio (TTR) & MTLD
  7.  Narrative-Density vs. Readability correlation (marker counts x FK grade)

Psycholinguistic grounding:
  - Levy (2008): surprisal(w) = -log2 P(w|context) ∝ reading time
  - Low-frequency / high-surprisal words → larger N400, longer eye fixations
  - High-density conspiracy markers + readability = cognitive load story

Output: figures/01_surprisal/ (PNG files)

Usage:
  python analysis_01_surprisal_lexical.py --data data/train_rehydrated.jsonl
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from wordfreq import zipf_frequency
import spacy
import textstat

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)

COLORS = {'Yes': '#e74c3c', 'No': '#3498db'}
FIG_DIR = Path('extra_analysis/figures/01_surprisal')
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            item = json.loads(line.strip())
            label = item.get('conspiracy', '').strip()
            if label in ('Yes', 'No'):
                data.append(item)
    return data


def get_text(item):
    return item.get('full_text', item.get('text', '')).strip()


def get_marker_count(item):
    return len(item.get('markers', []))


# ─────────────────────────────────────────────────────────────────────────────
# GPT-2 SURPRISAL
# ─────────────────────────────────────────────────────────────────────────────

def compute_surprisal(texts, device):
    """
    Returns a list of dicts per text:
      mean, variance, surprisals (array), positions (0→1), clause_boundary_surp, mid_clause_surp
    None if text is too short or error occurs.
    """
    print("  Loading GPT-2 (gpt2-medium for better estimates)...")
    try:
        model = GPT2LMHeadModel.from_pretrained('gpt2-medium').to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-medium')
    except Exception:
        print("  gpt2-medium unavailable, falling back to gpt2...")
        model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    model.eval()
    results = []

    # Try loading spaCy for clause boundary detection
    try:
        nlp_mini = spacy.load('en_core_web_sm', disable=['ner', 'lemmatizer'])
        use_spacy_clauses = True
    except Exception:
        use_spacy_clauses = False

    for text in tqdm(texts, desc="  GPT-2 surprisal", ncols=80):
        if not text or len(text.split()) < 5:
            results.append(None)
            continue
        try:
            enc = tokenizer(text, return_tensors='pt',
                            truncation=True, max_length=512).to(device)
            with torch.no_grad():
                logits = model(**enc).logits          # [1, T, V]
                log_p  = torch.log_softmax(logits, dim=-1)
                ids    = enc['input_ids'][0, 1:]       # target tokens
                preds  = log_p[0, :-1, :]              # predictions
                if ids.numel() == 0:
                    results.append(None)
                    continue
                token_surp = (-preds[range(len(ids)), ids] /
                              torch.log(torch.tensor(2.0))).cpu().numpy()

            positions = np.linspace(0, 1, len(token_surp))

            # Clause-boundary surprisal: tokens near punctuation subwords
            token_strs = tokenizer.convert_ids_to_tokens(ids.cpu().tolist())
            boundary_mask = np.array([
                any(p in (t or '') for p in (',', '.', ';', ':', '!', '?', 'Ġ,', 'Ġ.'))
                for t in token_strs
            ])
            cb_surp = float(token_surp[boundary_mask].mean()) if boundary_mask.any() else None
            mc_surp = float(token_surp[~boundary_mask].mean()) if (~boundary_mask).any() else None

            results.append({
                'mean':         float(np.mean(token_surp)),
                'variance':     float(np.var(token_surp)),
                'surprisals':   token_surp.tolist(),
                'positions':    positions.tolist(),
                'clause_boundary': cb_surp,
                'mid_clause':      mc_surp,
            })
        except Exception:
            results.append(None)

    del model
    if device != 'cpu':
        torch.cuda.empty_cache()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# LEXICAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV'}

def compute_lexical(texts, nlp):
    results = []
    for text in tqdm(texts, desc="  Lexical frequency", ncols=80):
        if not text:
            results.append(None)
            continue
        doc = nlp(text[:4000])
        freqs, low_freq = [], []
        for tok in doc:
            if tok.is_alpha and not tok.is_stop and tok.pos_ in CONTENT_POS:
                z = zipf_frequency(tok.lemma_.lower(), 'en', minimum=0.0)
                freqs.append(z)
                low_freq.append(z < 3.5)
        if not freqs:
            results.append(None)
            continue
        results.append({
            'mean_zipf':      float(np.mean(freqs)),
            'low_freq_ratio': float(np.mean(low_freq)),
            'n_content':      len(freqs),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# READABILITY + LEXICAL DIVERSITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_mtld(text, threshold=0.720):
    tokens = [t.lower() for t in text.split() if t.isalpha()]
    if len(tokens) < 10:
        return None
    def _one_pass(toks):
        factors, tc = 0, 0
        types = set()
        for w in toks:
            tc += 1
            types.add(w)
            if len(types) / tc <= threshold:
                factors += 1
                tc, types = 0, set()
        if tc > 0:
            rem = len(types) / tc if tc > 0 else 1.0
            factors += (1.0 - rem) / (1.0 - threshold)
        return len(toks) / factors if factors > 0 else float('nan')
    fwd = _one_pass(tokens)
    bwd = _one_pass(list(reversed(tokens)))
    if np.isnan(fwd) or np.isnan(bwd):
        return None
    return (fwd + bwd) / 2.0


def compute_readability(texts):
    results = []
    for text in tqdm(texts, desc="  Readability/TTR/MTLD", ncols=80):
        if not text or len(text.split()) < 10:
            results.append(None)
            continue
        toks = [t.lower() for t in text.split() if t.isalpha()]
        ttr  = len(set(toks)) / len(toks) if toks else None
        try:
            fk   = textstat.flesch_kincaid_grade(text)
            smog = textstat.smog_index(text) if len(text.split('.')) >= 3 else None
        except Exception:
            fk, smog = None, None
        mtld = compute_mtld(text)
        results.append({'fk': fk, 'smog': smog, 'ttr': ttr, 'mtld': mtld})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def _violin_pair(ax, yes_vals, no_vals, ylabel, title, show_p=True):
    yes_clean = [v for v in yes_vals if v is not None and np.isfinite(v)]
    no_clean  = [v for v in no_vals  if v is not None and np.isfinite(v)]
    if not yes_clean or not no_clean:
        ax.set_title(f'{title}\n(no data)')
        return
    parts = ax.violinplot([yes_clean, no_clean], positions=[1, 2],
                          widths=0.6, showmedians=True, showextrema=False)
    for pc, col in zip(parts['bodies'], [COLORS['Yes'], COLORS['No']]):
        pc.set_facecolor(col)
        pc.set_alpha(0.75)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Believer\n(Yes)', 'Non-Believer\n(No)'])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_p and len(yes_clean) > 1 and len(no_clean) > 1:
        _, p = stats.mannwhitneyu(yes_clean, no_clean, alternative='two-sided')
        star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(0.97, 0.96, f'p={p:.4f} {star}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
    # Print stats
    print(f"    {title}: Yes={np.mean(yes_clean):.3f}±{np.std(yes_clean):.3f}  "
          f"No={np.mean(no_clean):.3f}±{np.std(no_clean):.3f}")


def plot_surprisal(yes_s, no_s):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    _violin_pair(axes[0],
                 [s['mean'] for s in yes_s if s],
                 [s['mean'] for s in no_s  if s],
                 'Mean Surprisal (bits)', 'Mean Per-Token Surprisal')

    _violin_pair(axes[1],
                 [s['variance'] for s in yes_s if s],
                 [s['variance'] for s in no_s  if s],
                 'Surprisal Variance (bits²)', 'Surprisal Variance')

    # Clause boundary vs mid-clause
    ax = axes[2]
    cb_yes = [s['clause_boundary'] for s in yes_s if s and s['clause_boundary']]
    cb_no  = [s['clause_boundary'] for s in no_s  if s and s['clause_boundary']]
    mc_yes = [s['mid_clause']      for s in yes_s if s and s['mid_clause']]
    mc_no  = [s['mid_clause']      for s in no_s  if s and s['mid_clause']]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w/2,
           [np.mean(cb_yes), np.mean(mc_yes)],
           w, label='Believer (Yes)', color=COLORS['Yes'], alpha=0.8,
           yerr=[stats.sem(cb_yes), stats.sem(mc_yes)], capsize=4)
    ax.bar(x + w/2,
           [np.mean(cb_no), np.mean(mc_no)],
           w, label='Non-Believer (No)', color=COLORS['No'], alpha=0.8,
           yerr=[stats.sem(cb_no), stats.sem(mc_no)], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(['Clause Boundary', 'Mid-Clause'])
    ax.set_ylabel('Mean Surprisal (bits)')
    ax.set_title('Surprisal at Clause Boundaries')
    ax.legend(fontsize=9)

    fig.suptitle('GPT-2 Surprisal: Conspiracy vs. Non-Conspiracy Posts',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'surprisal_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: surprisal_distributions.png")


def plot_positional(yes_s, no_s, n_bins=10):
    fig, ax = plt.subplots(figsize=(11, 5))
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    for label, slist, color in [
        ('Believer (Yes)',     yes_s, COLORS['Yes']),
        ('Non-Believer (No)',  no_s,  COLORS['No']),
    ]:
        bin_means, bin_sems = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            vals = []
            for s in slist:
                if s is None:
                    continue
                surp = np.array(s['surprisals'])
                pos  = np.array(s['positions'])
                sel  = surp[(pos >= lo) & (pos < hi)]
                if len(sel):
                    vals.append(sel.mean())
            bin_means.append(np.mean(vals) if vals else np.nan)
            bin_sems.append(stats.sem(vals) if len(vals) > 1 else 0)

        bm = np.array(bin_means)
        bs = np.array(bin_sems)
        ax.plot(centers, bm, 'o-', color=color, label=label, lw=2.5)
        ax.fill_between(centers, bm - bs, bm + bs, alpha=0.15, color=color)

    ax.set_xlabel('Normalised Position in Text (beginning → end)')
    ax.set_ylabel('Mean Surprisal (bits)')
    ax.set_title('Surprisal Trajectory Across Text Position\n'
                 '(Levy 2008: surprisal ∝ reading time)')
    ax.set_xticks(centers)
    ax.set_xticklabels([f'{int(c*100)}%' for c in centers], rotation=30)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'surprisal_positional.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: surprisal_positional.png")


def plot_lexical_readability(yes_lex, no_lex, yes_rd, no_rd):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Lexical Processing Load: Conspiracy vs. Non-Conspiracy',
                 fontsize=14, y=1.01)

    _violin_pair(axes[0,0],
                 [f['mean_zipf'] for f in yes_lex if f],
                 [f['mean_zipf'] for f in no_lex  if f],
                 'Mean Zipf Frequency', 'Content-Word Lexical Frequency\n(lower = harder/rarer)')

    _violin_pair(axes[0,1],
                 [f['low_freq_ratio'] for f in yes_lex if f],
                 [f['low_freq_ratio'] for f in no_lex  if f],
                 'Fraction of Words', 'Low-Frequency Word Ratio\n(Zipf < 3.5 = "unusual" words)')

    _violin_pair(axes[0,2],
                 [f['fk']   for f in yes_rd if f and f['fk']   is not None],
                 [f['fk']   for f in no_rd  if f and f['fk']   is not None],
                 'Grade Level', 'Flesch-Kincaid Grade Level')

    _violin_pair(axes[1,0],
                 [f['smog'] for f in yes_rd if f and f['smog'] is not None],
                 [f['smog'] for f in no_rd  if f and f['smog'] is not None],
                 'SMOG Grade', 'SMOG Index\n(sentences needed ≥ 3)')

    _violin_pair(axes[1,1],
                 [f['ttr']  for f in yes_rd if f and f['ttr']  is not None],
                 [f['ttr']  for f in no_rd  if f and f['ttr']  is not None],
                 'Type-Token Ratio', 'Type-Token Ratio (TTR)\n(lexical variety)')

    _violin_pair(axes[1,2],
                 [f['mtld'] for f in yes_rd if f and f['mtld'] is not None],
                 [f['mtld'] for f in no_rd  if f and f['mtld'] is not None],
                 'MTLD Score', 'MTLD (Lexical Diversity)\n(higher = more diverse vocabulary)')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'lexical_readability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: lexical_readability.png")


def plot_narrative_density_vs_readability(yes_data, no_data, yes_rd, no_rd):
    """Scatter: marker count vs FK grade — does narrative density = harder text?"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, rd, color, label in [
        (axes[0], yes_data, yes_rd, COLORS['Yes'], 'Believers (Yes)'),
        (axes[1], no_data,  no_rd,  COLORS['No'],  'Non-Believers (No)'),
    ]:
        mc = [get_marker_count(d) for d, r in zip(data, rd) if r and r['fk'] is not None]
        fk = [r['fk'] for d, r in zip(data, rd) if r and r['fk'] is not None]
        if not mc:
            continue
        ax.scatter(mc, fk, alpha=0.3, color=color, s=20, edgecolors='none')
        if len(mc) > 5:
            m, b, r, p, _ = stats.linregress(mc, fk)
            xs = np.array([min(mc), max(mc)])
            ax.plot(xs, m*xs + b, 'k--', lw=1.5, label=f'r={r:.3f}, p={p:.3f}')
            ax.legend(fontsize=9)
        ax.set_xlabel('Narrative Density (marker count)')
        ax.set_ylabel('Flesch-Kincaid Grade Level')
        ax.set_title(f'Narrative Density vs. Readability\n{label}')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'narrative_density_vs_readability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: narrative_density_vs_readability.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/train_rehydrated.jsonl',
                        help='Path to train JSONL file')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU even if GPU is available')
    args = parser.parse_args()

    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("ANALYSIS 1: SURPRISAL & LEXICAL PROCESSING LOAD")
    print(f"{'='*60}")
    print(f"Device: {device}  |  Figures → {FIG_DIR}/")

    print("\n[Loading data]")
    data = load_data(args.data)
    yes_data = [d for d in data if d['conspiracy'] == 'Yes']
    no_data  = [d for d in data if d['conspiracy'] == 'No']
    print(f"  Yes: {len(yes_data)}  |  No: {len(no_data)}")

    yes_texts = [get_text(d) for d in yes_data]
    no_texts  = [get_text(d) for d in no_data]

    # ── 1. Surprisal ──────────────────────────────────────────────────────────
    print("\n[1/4] GPT-2 Surprisal")
    yes_surp = compute_surprisal(yes_texts, device)
    no_surp  = compute_surprisal(no_texts,  device)

    print("\n[2/4] spaCy for lexical features")
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])

    # ── 2. Lexical frequency ─────────────────────────────────────────────────
    yes_lex = compute_lexical(yes_texts, nlp)
    no_lex  = compute_lexical(no_texts,  nlp)

    # ── 3. Readability + diversity ───────────────────────────────────────────
    print("\n[3/4] Readability / TTR / MTLD")
    yes_rd = compute_readability(yes_texts)
    no_rd  = compute_readability(no_texts)

    # ── 4. Plots ─────────────────────────────────────────────────────────────
    print("\n[4/4] Generating figures")
    plot_surprisal(yes_surp, no_surp)
    plot_positional(yes_surp, no_surp)
    plot_lexical_readability(yes_lex, no_lex, yes_rd, no_rd)
    plot_narrative_density_vs_readability(yes_data, no_data, yes_rd, no_rd)

    print(f"\n✓ All figures saved to {FIG_DIR}/")


if __name__ == '__main__':
    main()
