#!/usr/bin/env python3
"""
Module A: RSA Surprisal Classifier
====================================
Implements the Rational Speech Act (RSA) framework as a zero-shot
Bayesian belief detector. No fine-tuning required.

Core idea (Goodman & Frank 2016):
  P(Belief | text) ∝ P(text | Belief) × P(Belief)

  P(text | Belief=Yes) = exp(-PPL(text | believer_prefix))
  P(text | Belief=No)  = exp(-PPL(text | skeptic_prefix))
  P(Belief=Yes)        = base rate from training data

This replaces the black-box MLP classifier with an interpretable
Bayesian update. The gap between this F1 and DeBERTa's F1 quantifies
exactly how much learned narrative structure adds beyond rational
pragmatic inference.

Additional analyses:
  1. RSA posterior distribution by true label (violin)
  2. Prior sensitivity: how does F1 change as we vary P(Belief=Yes)?
  3. RSA calibration curve vs DeBERTa calibration
  4. KL divergence between conditional distributions
  5. "Can't Tell" posterior analysis - why do humans find posts ambiguous?

Output: figures/rsa/ + results/rsa_predictions.csv

Usage:
  python bayesian_A_rsa.py --train data/train_rehydrated.jsonl
                           --dev   data/dev_public.jsonl
  python bayesian_A_rsa.py --train data/train_rehydrated.jsonl \
                           --dev   data/dev_public.jsonl --cpu
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
from scipy.special import softmax
from sklearn.metrics import (f1_score, accuracy_score,
                             classification_report, brier_score_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.calibration import calibration_curve
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm
import pandas as pd

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=1.1)

COLORS = {'Yes': '#e74c3c', 'No': '#3498db', "Can't tell": '#95a5a6'}
FIG_DIR = Path('figures/rsa')
RES_DIR = Path('results')
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

# ── RSA CONTEXT PREFIXES ─────────────────────────────────────────────────────
# These operationalise S1(utterance | belief_state) — what would a
# pragmatic speaker who holds this belief state say?

BELIEVER_PREFIX = (
    "The following text is written by someone who genuinely believes "
    "a conspiracy theory is real and is trying to convince others of it: "
)
SKEPTIC_PREFIX = (
    "The following text is written by someone who is critically analyzing, "
    "debunking, or discussing a conspiracy theory without personally believing it: "
)
NEUTRAL_PREFIX = ""   # baseline: no conditioning


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path, include_canttell=False):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            lbl = d.get('conspiracy', '').strip()
            if lbl in ('Yes', 'No'):
                rows.append(d)
            elif lbl == "Can't tell" and include_canttell:
                rows.append(d)
    return rows


def get_text(d):
    return d.get('full_text', d.get('text', '')).strip()


def get_marker_count(d):
    return len(d.get('markers', []))


# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL PERPLEXITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_conditional_ppl(text, prefix, model, tokenizer, device,
                             max_target_tokens=256, stride=128):
    """
    Compute perplexity of `text` conditioned on `prefix`.

    PPL(text | prefix) = exp(-1/N * sum_i log P(text_i | prefix + text_<i))

    Uses sliding-window approach for long texts.
    Only the `text` tokens contribute to the loss (prefix tokens are
    treated as context only — they do not appear in the denominator).
    """
    full = prefix + text
    enc_full   = tokenizer(full,   return_tensors='pt',
                           add_special_tokens=True)
    enc_prefix = tokenizer(prefix, return_tensors='pt',
                           add_special_tokens=False)

    full_ids   = enc_full['input_ids'][0]
    n_prefix   = len(enc_prefix['input_ids'][0])

    # Truncate: keep all prefix tokens + up to max_target_tokens
    max_len    = n_prefix + max_target_tokens
    full_ids   = full_ids[:max_len]
    n_target   = len(full_ids) - n_prefix

    if n_target <= 0:
        return float('nan')

    full_ids = full_ids.unsqueeze(0).to(device)

    with torch.no_grad():
        out    = model(input_ids=full_ids, labels=full_ids)
        logits = out.logits[0]   # [T, V]

    # Compute token-level NLL only for target tokens (not prefix)
    log_probs = torch.log_softmax(logits, dim=-1)  # [T, V]
    target    = full_ids[0, 1:]                    # shifted targets [T-1]
    token_nlls = -log_probs[:-1, :].gather(1, target.unsqueeze(1)).squeeze(1)

    # Only count target portion (indices n_prefix-1 onward in shifted array)
    target_nlls = token_nlls[n_prefix - 1:]
    if len(target_nlls) == 0:
        return float('nan')

    return float(target_nlls.mean().exp().item())


# ─────────────────────────────────────────────────────────────────────────────
# BATCH RSA INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def run_rsa_inference(data, model, tokenizer, device,
                      prior_yes=0.431, max_tokens=256):
    """
    For each post compute:
      - PPL under believer prefix
      - PPL under skeptic prefix
      - PPL under neutral prefix (baseline)
      - RSA posterior P(Yes | text)
      - Prediction at threshold 0.5

    Returns DataFrame with one row per post.
    """
    records = []

    for d in tqdm(data, desc='RSA inference', ncols=80):
        text  = get_text(d)
        label = d.get('conspiracy', '').strip()

        if not text or len(text.split()) < 5:
            continue

        ppl_yes  = compute_conditional_ppl(
            text, BELIEVER_PREFIX, model, tokenizer, device, max_tokens)
        ppl_no   = compute_conditional_ppl(
            text, SKEPTIC_PREFIX,  model, tokenizer, device, max_tokens)
        ppl_base = compute_conditional_ppl(
            text, NEUTRAL_PREFIX,  model, tokenizer, device, max_tokens)

        if np.isnan(ppl_yes) or np.isnan(ppl_no):
            continue

        # Likelihood: lower perplexity = higher probability
        # P(text | belief) ∝ exp(-PPL) — using log-sum-exp for stability
        log_lik_yes = -ppl_yes
        log_lik_no  = -ppl_no

        # Log posterior (unnormalised)
        log_post_yes = log_lik_yes + np.log(prior_yes)
        log_post_no  = log_lik_no  + np.log(1 - prior_yes)

        # Normalise via softmax
        posteriors = softmax([log_post_yes, log_post_no])
        p_yes = float(posteriors[0])

        # Likelihood ratio (how many times more likely under believer model)
        llr = ppl_no / (ppl_yes + 1e-9)

        records.append({
            'id':          d.get('_id', ''),
            'label':       label,
            'text_len':    len(text.split()),
            'marker_count': get_marker_count(d),
            'ppl_believer': ppl_yes,
            'ppl_skeptic':  ppl_no,
            'ppl_neutral':  ppl_base,
            'llr':          llr,
            'p_yes_rsa':    p_yes,
            'pred_rsa':     'Yes' if p_yes >= 0.5 else 'No',
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_posterior_distributions(df):
    """Violin: RSA posterior P(Yes) separated by true label."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'RSA Posterior P(Belief=Yes | text)\n'
        'Rational Speech Act inference — zero-shot, no fine-tuning',
        fontsize=13, y=1.02
    )

    # Left: violin by true label
    ax = axes[0]
    bin_labels = [l for l in ('Yes', 'No') if l in df['label'].values]
    groups = [df[df['label'] == l]['p_yes_rsa'].dropna().values
              for l in bin_labels]
    vp = ax.violinplot(groups, positions=range(1, len(bin_labels)+1),
                       widths=0.6, showmedians=True, showextrema=False)
    for pc, lbl in zip(vp['bodies'], bin_labels):
        pc.set_facecolor(COLORS.get(lbl, '#aaa'))
        pc.set_alpha(0.75)
    ax.set_xticks(range(1, len(bin_labels)+1))
    ax.set_xticklabels([f'True: {l}' for l in bin_labels])
    ax.axhline(0.5, color='black', lw=1.2, ls='--', label='Decision boundary')
    ax.set_ylabel('RSA Posterior P(Yes | text)')
    ax.set_title('Posterior by True Label')
    ax.legend(fontsize=9)

    # Print stats
    for lbl, grp in zip(bin_labels, groups):
        print(f'  {lbl}: median={np.median(grp):.3f}  '
              f'mean={np.mean(grp):.3f}  '
              f'above-0.5={np.mean(grp>0.5)*100:.1f}%')

    # Right: posterior vs marker count scatter
    ax = axes[1]
    for lbl in bin_labels:
        sub = df[df['label'] == lbl]
        ax.scatter(sub['marker_count'], sub['p_yes_rsa'],
                   alpha=0.25, s=15, color=COLORS.get(lbl, '#aaa'),
                   edgecolors='none', label=lbl)
        if len(sub) > 5:
            m, b, r, p, _ = stats.linregress(
                sub['marker_count'], sub['p_yes_rsa'])
            xs = np.array([sub['marker_count'].min(),
                           sub['marker_count'].max()])
            ax.plot(xs, m*xs+b, '--', color=COLORS.get(lbl, '#aaa'),
                    lw=2, label=f'{lbl} r={r:.3f}')
    ax.axhline(0.5, color='black', lw=1.2, ls='--')
    ax.set_xlabel('Narrative Marker Count')
    ax.set_ylabel('RSA Posterior P(Yes | text)')
    ax.set_title('RSA Posterior vs. Narrative Density')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'rsa_posterior_distributions.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: rsa_posterior_distributions.png')


def plot_prior_sensitivity(df):
    """
    Vary prior P(Yes) from 0.05 to 0.95 and plot F1 / accuracy.
    Shows how much the RSA model depends on the assumed base rate.
    """
    priors  = np.linspace(0.05, 0.95, 40)
    f1s_yes = []
    f1s_no  = []
    accs    = []

    df_eval = df[df['label'].isin(['Yes', 'No'])].copy()
    y_true  = (df_eval['label'] == 'Yes').astype(int).values

    for prior in priors:
        log_lik_yes = -df_eval['ppl_believer'].values
        log_lik_no  = -df_eval['ppl_skeptic'].values
        log_post_yes = log_lik_yes + np.log(prior)
        log_post_no  = log_lik_no  + np.log(1 - prior)
        # Element-wise softmax
        stack = np.stack([log_post_yes, log_post_no], axis=1)
        p_yes = softmax(stack, axis=1)[:, 0]
        preds = (p_yes >= 0.5).astype(int)

        f1s_yes.append(f1_score(y_true, preds, pos_label=1,
                                zero_division=0))
        f1s_no.append(f1_score(y_true, preds, pos_label=0,
                               zero_division=0))
        accs.append(accuracy_score(y_true, preds))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(priors, f1s_yes, 'o-', color=COLORS['Yes'],
            lw=2.5, ms=4, label='F1 — Believer class (Yes)')
    ax.plot(priors, f1s_no,  's-', color=COLORS['No'],
            lw=2.5, ms=4, label='F1 — Non-Believer class (No)')
    ax.plot(priors, accs,    '^--', color='gray',
            lw=1.5, ms=4, label='Accuracy')

    # Mark the true base rate
    base = 0.431
    ax.axvline(base, color='black', lw=1.2, ls=':', label=f'True base rate ({base})')
    ax.set_xlabel('Prior P(Belief = Yes)', fontsize=11)
    ax.set_ylabel('Performance')
    ax.set_title('RSA Prior Sensitivity Analysis\n'
                 '(how much does the assumed base rate matter?)',
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0.02, 0.98)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'rsa_prior_sensitivity.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: rsa_prior_sensitivity.png')


def plot_ppl_gap(df):
    """
    Key plot: PPL(skeptic) - PPL(believer) per post.
    Positive = text is more likely under believer model.
    This is the most interpretable single quantity from RSA.
    """
    df = df.copy()
    df['ppl_gap'] = df['ppl_skeptic'] - df['ppl_believer']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        'Perplexity Gap: PPL(skeptic context) - PPL(believer context)\n'
        'Positive = text more probable under believer speaker model',
        fontsize=12, y=1.02
    )

    # Left: violin by label
    ax = axes[0]
    groups = [df[df['label'] == l]['ppl_gap'].dropna().values
              for l in ('Yes', 'No')]
    vp = ax.violinplot(groups, positions=[1, 2], widths=0.55,
                       showmedians=True, showextrema=False)
    for pc, col in zip(vp['bodies'], [COLORS['Yes'], COLORS['No']]):
        pc.set_facecolor(col); pc.set_alpha(0.75)
    ax.axhline(0, color='black', lw=1.0, ls='--')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Believers\n(Yes)', 'Non-Believers\n(No)'])
    ax.set_ylabel('PPL Gap (bits)')
    ax.set_title('PPL Gap by True Label')

    y_v, n_v = groups
    _, p = stats.mannwhitneyu(y_v, n_v, alternative='two-sided')
    ax.text(0.97, 0.96, f'p={p:.4f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
    print(f'  PPL gap: Yes median={np.median(y_v):.2f}  '
          f'No median={np.median(n_v):.2f}  p={p:.4f}')

    # Right: scatter PPL gap vs marker count
    ax = axes[1]
    for lbl in ('Yes', 'No'):
        sub = df[df['label'] == lbl]
        ax.scatter(sub['marker_count'], sub['ppl_gap'],
                   alpha=0.2, s=15, color=COLORS[lbl],
                   edgecolors='none', label=lbl)
        if len(sub) > 5:
            m, b, r, p_val, _ = stats.linregress(
                sub['marker_count'], sub['ppl_gap'])
            xs = np.array([sub['marker_count'].min(),
                           sub['marker_count'].max()])
            ax.plot(xs, m*xs+b, '--', color=COLORS[lbl], lw=2,
                    label=f'{lbl} r={r:.3f} p={p_val:.3f}')

    ax.axhline(0, color='black', lw=1.0, ls='--')
    ax.set_xlabel('Narrative Marker Count')
    ax.set_ylabel('PPL Gap')
    ax.set_title('PPL Gap vs. Narrative Density\n'
                 '(do denser narratives look more "believer-like" to GPT-2?)')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'rsa_ppl_gap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: rsa_ppl_gap.png')


def plot_calibration(df):
    """Calibration curve: how well-calibrated is the RSA posterior?"""
    df_eval = df[df['label'].isin(['Yes', 'No'])].copy()
    y_true  = (df_eval['label'] == 'Yes').astype(int).values
    y_prob  = df_eval['p_yes_rsa'].values

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    brier = brier_score_loss(y_true, y_prob)
    auc   = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(prob_pred, prob_true, 'o-', color=COLORS['Yes'],
            lw=2.5, ms=7, label=f'RSA (Brier={brier:.3f}, AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration')
    ax.fill_between([0, 1], [0, 1], [0, 1],
                    alpha=0.05, color='gray')
    ax.set_xlabel('Mean Predicted Probability (RSA posterior)')
    ax.set_ylabel('Fraction of True Positives')
    ax.set_title('RSA Calibration Curve\n'
                 '(Bayesian posterior — no post-hoc calibration)',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'rsa_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: rsa_calibration.png  '
          f'(Brier={brier:.3f}, AUC={auc:.3f})')


def plot_cant_tell_posteriors(df_all, df_rsa):
    """
    For "Can't Tell" posts: what does the RSA posterior look like?
    If RSA produces flat posteriors for exactly these posts,
    the model's uncertainty mirrors human annotator confusion.
    """
    ct_ids = df_all[df_all['label'] == "Can't tell"]['id'].values
    binary_ids = df_all[df_all['label'].isin(['Yes', 'No'])]['id'].values

    ct_preds = df_rsa[df_rsa['id'].isin(ct_ids)]['p_yes_rsa'].dropna()
    bi_preds = df_rsa[df_rsa['id'].isin(binary_ids)]['p_yes_rsa'].dropna()

    if len(ct_preds) < 5:
        print("  Not enough Can't Tell posts in RSA output to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "RSA Posteriors for 'Can't Tell' Posts\n"
        "If human annotators are confused for the right reason, "
        "RSA posteriors should cluster near 0.5",
        fontsize=12, y=1.02
    )

    ax = axes[0]
    ax.hist(ct_preds, bins=20, color=COLORS["Can't tell"],
            alpha=0.8, density=True, label="Can't Tell (n={})".format(len(ct_preds)))
    ax.hist(bi_preds[bi_preds >= 0.4][bi_preds[bi_preds >= 0.4] <= 0.6],
            bins=20, color='lightblue', alpha=0.5, density=True,
            label='Binary posts near 0.5')
    ax.axvline(0.5, color='black', lw=1.5, ls='--', label='Decision boundary')
    ax.set_xlabel("RSA Posterior P(Yes | text)")
    ax.set_ylabel("Density")
    ax.set_title("Can't Tell: Posterior Distribution")
    ax.legend(fontsize=9)

    # Entropy of posterior as uncertainty measure
    ax = axes[1]
    def entropy(p):
        p = np.clip(p, 1e-9, 1-1e-9)
        return -(p * np.log2(p) + (1-p) * np.log2(1-p))

    ct_ent  = entropy(ct_preds.values)
    yes_preds = df_rsa[df_rsa['label'] == 'Yes']['p_yes_rsa'].dropna()
    no_preds  = df_rsa[df_rsa['label'] == 'No']['p_yes_rsa'].dropna()
    yes_ent = entropy(yes_preds.values)
    no_ent  = entropy(no_preds.values)

    vp = ax.violinplot([yes_ent, no_ent, ct_ent],
                       positions=[1, 2, 3], widths=0.55,
                       showmedians=True, showextrema=False)
    for pc, col in zip(vp['bodies'],
                       [COLORS['Yes'], COLORS['No'],
                        COLORS["Can't tell"]]):
        pc.set_facecolor(col); pc.set_alpha(0.75)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Believers\n(Yes)', 'Non-Believers\n(No)',
                        "Can't Tell"])
    ax.set_ylabel('Posterior Entropy (bits)')
    ax.set_title("Model Uncertainty (Entropy) by Label\n"
                 "Higher entropy = model is less certain")

    print(f"  Can't Tell median entropy: {np.median(ct_ent):.3f}")
    print(f"  Yes median entropy:        {np.median(yes_ent):.3f}")
    print(f"  No  median entropy:        {np.median(no_ent):.3f}")

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'rsa_canttell.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: rsa_canttell.png")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_rsa(df, label='dev'):
    df_eval = df[df['label'].isin(['Yes', 'No'])].copy()
    y_true  = df_eval['label'].values
    y_pred  = df_eval['pred_rsa'].values

    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_yes   = f1_score(y_true, y_pred, pos_label='Yes', average='binary',
                        zero_division=0)
    acc      = accuracy_score(y_true, y_pred)

    print(f'\n{"="*50}')
    print(f'RSA CLASSIFIER EVALUATION ({label})')
    print(f'{"="*50}')
    print(f'  Macro F1:  {f1_macro:.4f}')
    print(f'  F1 (Yes):  {f1_yes:.4f}')
    print(f'  Accuracy:  {acc:.4f}')
    print()
    print(classification_report(y_true, y_pred, zero_division=0))

    return f1_macro


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='data/psycomark_train.jsonl')
    parser.add_argument('--dev',   default='data/psycomark_dev.jsonl')
    parser.add_argument('--model', default='gpt2-medium',
                        help='GPT-2 model name. gpt2/gpt2-medium/gpt2-large')
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--dev-only', action='store_true',
                        help='Run dev inference only (skip training)')
    args = parser.parse_args()

    device = 'cpu' if args.cpu else (
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*60}')
    print('MODULE A: RSA SURPRISAL CLASSIFIER')
    print(f'{"="*60}')
    print(f'Model: {args.model}  |  Device: {device}')
    if args.dev_only:
        print('Mode: DEV-ONLY (skipping training)')

    # ── Data ────────────────────────────────────────────────────────────────
    print('\n[1/5] Loading data...')
    train_data = load_data(args.train, include_canttell=True)
    dev_data   = load_data(args.dev,   include_canttell=True)

    # Compute base rate from binary-labelled training posts only
    binary_train = [d for d in train_data
                    if d['conspiracy'] in ('Yes', 'No')]
    prior_yes = sum(1 for d in binary_train
                    if d['conspiracy'] == 'Yes') / len(binary_train)
    print(f'  Base rate P(Yes) = {prior_yes:.4f}  '
          f'(from {len(binary_train)} binary-labelled posts)')

    # ── Model ────────────────────────────────────────────────────────────────
    print(f'\n[2/5] Loading {args.model}...')
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    model     = GPT2LMHeadModel.from_pretrained(args.model).to(device)
    model.eval()
    print(f'  Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # ── RSA inference ────────────────────────────────────────────────────────
    if not args.dev_only:
        print('\n[3/5] Running RSA inference on training data...')
        df_train_all = pd.DataFrame([
            {'id': d.get('_id',''), 'label': d['conspiracy']} for d in train_data
        ])

        df_train = run_rsa_inference(
            [d for d in train_data if d['conspiracy'] in ('Yes','No')],
            model, tokenizer, device,
            prior_yes=prior_yes, max_tokens=args.max_tokens
        )

        print('\n[4/5] Running RSA inference on dev data...')
    else:
        print('\n[3/5] Skipping training inference...')
        df_train = None
        print('\n[4/5] Running RSA inference on dev data...')
    
    df_dev = run_rsa_inference(
        dev_data, model, tokenizer, device,
        prior_yes=prior_yes, max_tokens=args.max_tokens
    )

    # ── Evaluation ───────────────────────────────────────────────────────────
    print('\n[5/5] Evaluation and figures...')
    if not args.dev_only:
        evaluate_rsa(df_train, 'train')
    evaluate_rsa(df_dev,   'dev')

    # Save predictions
    if not args.dev_only:
        df_train.to_csv(RES_DIR / 'rsa_train_predictions.csv', index=False)
    df_dev.to_csv(  RES_DIR / 'rsa_dev_predictions.csv',   index=False)
    print(f'  Predictions saved to {RES_DIR}/')

    # ── Plots ─────────────────────────────────────────────────────────────────
    print('\nGenerating figures...')
    
    if not args.dev_only:
        # Use training data (larger N) for distribution plots
        plot_posterior_distributions(df_train)
        plot_ppl_gap(df_train)
        plot_prior_sensitivity(df_train)
        plot_calibration(df_train)

        # "Can't Tell" analysis on full training data
        df_all_with_ct = run_rsa_inference(
            [d for d in train_data if d['conspiracy'] == "Can't tell"],
            model, tokenizer, device,
            prior_yes=prior_yes, max_tokens=args.max_tokens
        )
        df_all_for_ct = pd.concat([
            df_train,
            df_all_with_ct
        ], ignore_index=True)

        df_all_label = pd.DataFrame([
            {'id': d.get('_id',''), 'label': d['conspiracy']}
            for d in train_data
        ])
        plot_cant_tell_posteriors(df_all_label, df_all_for_ct)
        df_all_for_ct.to_csv(RES_DIR / 'rsa_train_predictions.csv', index=False)

        print(f'\n✓ All figures saved to {FIG_DIR}/')
        print(f'✓ Predictions saved to {RES_DIR}/')

        # Print summary comparison table
        print('\n' + '='*50)
        print('COMPARISON TABLE (for your presentation)')
        print('='*50)
        print(f'  DeBERTa ensemble (official test):  F1 = 0.750')
        print(f'  DeBERTa ensemble (5-fold CV):      F1 = 0.734')
        f1_rsa = evaluate_rsa.__doc__  # placeholder
        df_e = df_train[df_train['label'].isin(['Yes','No'])]
        f1   = f1_score(df_e['label'].values, df_e['pred_rsa'].values,
                        average='macro', zero_division=0)
        print(f'  RSA surprisal (zero-shot):         F1 = {f1:.3f}')
        print(f'\n  Gap (DeBERTa - RSA) = {0.750 - f1:.3f} F1 points')
        print(f'  This gap = value of learned narrative structure')
        print(f'  beyond what rational Bayesian inference alone can recover.')
    else:
        # Dev-only mode: Generate dev-specific figures
        plot_posterior_distributions(df_dev)
        plot_ppl_gap(df_dev)
        plot_calibration(df_dev)
        print(f'\n✓ Dev figures saved to {FIG_DIR}/')
        print(f'✓ Dev predictions saved to {RES_DIR}/rsa_dev_predictions.csv')


if __name__ == '__main__':
    main()