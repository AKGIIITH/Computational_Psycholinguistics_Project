#!/usr/bin/env python3
"""
Analysis 5: DeBERTa Interpretability
=====================================
Two complementary interpretability methods for the trained DeBERTa classifier.

Method A — Gradient x Input attribution (fast, per-token):
  - Computes which tokens drive each class prediction
  - Aggregates top tokens across many examples
  - Checks whether Actor/Action/Effect/Evidence/Victim spans
    receive disproportionately high attribution (validates narrative density thesis)

Method B — Attention Visualisation:
  - Extracts attention weights from last 2 DeBERTa layers, averaged over heads
  - Plots per-token attention heatmaps for selected examples

GPU notes:
  - GTX 1650 (4GB): use --model-name distilbert-base-uncased --quick
  - RTX 6000 (48GB): use --model-name microsoft/deberta-v3-large --model-path <ckpt>

Output: figures/05_interpretability/

Usage:
  # Quick / local test (GTX 1650 safe):
  python analysis_05_interpretability.py --data data/train_rehydrated.jsonl --quick

  # Full model with saved checkpoint (RTX 6000):
  python analysis_05_interpretability.py \
      --data data/train_rehydrated.jsonl \
      --model-path subtask2/models/improved_seed2026.pt \
      --model-name microsoft/deberta-v3-large
"""

import argparse
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=1.0)

COLORS = {'Yes': '#e74c3c', 'No': '#3498db'}
MARKER_COLORS = {
    'Actor':    '#e74c3c',
    'Action':   '#e67e22',
    'Effect':   '#f1c40f',
    'Evidence': '#2ecc71',
    'Victim':   '#9b59b6',
}
FIG_DIR = Path('figures/05_interpretability')
FIG_DIR.mkdir(parents=True, exist_ok=True)
LABEL2ID = {'No': 0, 'Yes': 1}


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath, canttell_as_yes=True):
    data = []
    with open(filepath) as f:
        for line in f:
            item = json.loads(line.strip())
            lbl  = item.get('conspiracy', '').strip()
            if lbl.lower() == "can't tell":
                item['conspiracy'] = 'Yes' if canttell_as_yes else None
            if item.get('conspiracy') not in ('Yes', 'No'):
                continue
            data.append(item)
    return data


def get_text(item):
    return item.get('full_text', item.get('text', '')).strip()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL (mirrors train_improved.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

class ConspiracyClassifier(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name,
            output_attentions=True,
            output_hidden_states=True,
            ignore_mismatched_sizes=True,
        )
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2),
        )

    def forward(self, input_ids, attention_mask):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.classifier(pooled), out.attentions


class SimpleDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data, self.tokenizer, self.max_length = data, tokenizer, max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item  = self.data[idx]
        text  = get_text(item)
        label = LABEL2ID.get(item['conspiracy'], 0)
        enc   = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding='max_length', return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          label,
        }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def quick_train(data, model_name, device, epochs=3, lr=2e-5,
                max_length=128, batch_size=16):
    print(f"  Quick-training {model_name} for {epochs} epochs...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = ConspiracyClassifier(model_name).to(device)

    # Unfreeze only top 4 encoder layers
    for p in model.encoder.parameters():
        p.requires_grad = False
    layers = []
    if hasattr(model.encoder, 'encoder') and hasattr(model.encoder.encoder, 'layer'):
        layers = model.encoder.encoder.layer
    elif hasattr(model.encoder, 'transformer'):
        layers = model.encoder.transformer.layer
    for layer in layers[-4:]:
        for p in layer.parameters():
            p.requires_grad = True

    loader = DataLoader(
        SimpleDataset(data, tokenizer, max_length),
        batch_size=batch_size, shuffle=True, num_workers=2,
    )
    opt      = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    total    = len(loader) * epochs
    sched    = get_cosine_schedule_with_warmup(opt, total // 10, total)
    crit     = nn.CrossEntropyLoss(label_smoothing=0.1)

    model.train()
    for ep in range(epochs):
        correct = total_n = 0
        for batch in tqdm(loader, desc=f"  Epoch {ep+1}/{epochs}", ncols=80):
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            lbls  = batch['label'].to(device)
            opt.zero_grad()
            logits, _ = model(ids, mask)
            loss = crit(logits, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            correct += (logits.argmax(-1) == lbls).sum().item()
            total_n += len(lbls)
        print(f"    Acc: {correct/total_n:.4f}")

    return model, tokenizer


def load_checkpoint(path, model_name, device):
    print(f"  Loading checkpoint: {path}")
    ckpt  = torch.load(path, map_location=device)
    model = ConspiracyClassifier(model_name).to(device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# GRADIENT × INPUT ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def get_attribution(model, tokenizer, text, target_class, device,
                    max_length=128):
    """
    Gradient × Input attribution for each subword token.
    Returns: (token_strings, attribution_scores)
    """
    model.eval()
    enc = tokenizer(
        text, truncation=True, max_length=max_length,
        padding='max_length', return_tensors='pt',
    )
    input_ids = enc['input_ids'].to(device)
    attn_mask = enc['attention_mask'].to(device)

    # Get the embedding module (works for BERT, DeBERTa, DistilBERT)
    embed_mod = None
    for name, mod in model.encoder.named_modules():
        if isinstance(mod, nn.Embedding) and mod.weight.shape[0] > 100:
            embed_mod = mod
            break
    if embed_mod is None:
        return None, None

    # Enable grad on embeddings
    embeds = embed_mod(input_ids).detach().requires_grad_(True)

    # Forward hook to replace embedding output
    handle = embed_mod.register_forward_hook(
        lambda m, i, o: embeds
    )
    try:
        with torch.enable_grad():
            logits, _ = model(input_ids, attn_mask)
            logit = logits[0, target_class]
            logit.backward()
    finally:
        handle.remove()

    if embeds.grad is None:
        return None, None

    scores = (embeds.grad * embeds).norm(dim=-1).squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    real   = attn_mask[0].cpu().numpy().astype(bool)
    return [t for t, m in zip(tokens, real) if m], scores[real]


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE TOKEN ATTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_top_tokens(data, model, tokenizer, device,
                          n_samples=200, max_length=128, top_n=25):
    token_attr = defaultdict(list)
    sample = (
        [d for d in data if d['conspiracy'] == 'Yes'][:n_samples//2] +
        [d for d in data if d['conspiracy'] == 'No'][:n_samples//2]
    )
    for item in tqdm(sample, desc="  Aggregating attributions", ncols=80):
        label  = LABEL2ID[item['conspiracy']]
        toks, scores = get_attribution(
            model, tokenizer, get_text(item), label, device, max_length
        )
        if toks is None:
            continue
        for t, s in zip(toks, scores):
            clean = t.replace('▁','').replace('Ġ','').replace('##','').lower()
            if len(clean) >= 2 and clean.isalpha():
                token_attr[clean].append(float(s))

    ranked = sorted(
        {t: np.mean(v) for t, v in token_attr.items() if len(v) >= 3}.items(),
        key=lambda x: x[1], reverse=True
    )
    return ranked[:top_n], ranked[-top_n:][::-1]


# ─────────────────────────────────────────────────────────────────────────────
# MARKER SPAN ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def analyse_marker_attributions(data, model, tokenizer, device,
                                 n_samples=150, max_length=128):
    inside  = defaultdict(list)
    outside = []
    ratio   = defaultdict(list)

    sample = [d for d in data if d.get('markers')][:n_samples]

    for item in tqdm(sample, desc="  Marker span attribution", ncols=80):
        text  = get_text(item)
        label = LABEL2ID[item['conspiracy']]
        toks, scores = get_attribution(
            model, tokenizer, text, label, device, max_length
        )
        if toks is None:
            continue

        enc     = tokenizer(text, truncation=True, max_length=max_length,
                            return_offsets_mapping=True, padding=False)
        offsets = enc['offset_mapping']
        n_real  = min(len(offsets), len(scores))
        in_span = np.zeros(n_real, dtype=bool)

        for marker in item.get('markers', []):
            mtype  = marker.get('type', '')
            ms, me = marker.get('startIndex', 0), marker.get('endIndex', 0)
            span_sc = []
            for ti, (cs, ce) in enumerate(offsets[:n_real]):
                if cs >= ms and ce <= me + 3:
                    in_span[ti] = True
                    span_sc.append(scores[ti])
            if span_sc:
                inside[mtype].extend(span_sc)

        out_sc = scores[:n_real][~in_span]
        outside.extend(out_sc.tolist())

        for mtype, msc in inside.items():
            ratio[mtype].append(np.mean(msc) / (np.mean(out_sc) + 1e-9))

    return inside, outside, ratio


# ─────────────────────────────────────────────────────────────────────────────
# ATTENTION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def get_attention_map(model, tokenizer, text, device, max_length=128):
    enc = tokenizer(text, truncation=True, max_length=max_length,
                    padding=False, return_tensors='pt')
    ids  = enc['input_ids'].to(device)
    mask = enc['attention_mask'].to(device)
    model.eval()
    with torch.no_grad():
        _, attns = model(ids, mask)
    if not attns:
        return None, None
    stacked = torch.stack(list(attns[-2:]), 0)         # last 2 layers
    avg     = stacked.mean(dim=(0, 2)).squeeze(0)      # [T, T]
    n       = mask[0].sum().item()
    tokens  = tokenizer.convert_ids_to_tokens(ids[0].cpu().tolist())
    return avg[:n, :n].cpu().numpy(), tokens[:n]


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def _clean(t):
    return t.replace('▁','').replace('Ġ','').replace('##','') or t


def plot_top_tokens(top_pos, top_neg):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('DeBERTa Token Attribution (Gradient × Input)\n'
                 'Top tokens driving each class prediction',
                 fontsize=13, y=1.02)

    for ax, ranked, title, color in [
        (axes[0], top_pos, 'Tokens → Believer (Yes)',     COLORS['Yes']),
        (axes[1], top_neg, 'Tokens → Non-Believer (No)',  COLORS['No']),
    ]:
        toks = [_clean(t) for t, _ in ranked]
        vals = [v for _, v in ranked]
        y    = np.arange(len(toks))
        ax.barh(y, vals, color=color, alpha=0.8)
        ax.set_yticks(y); ax.set_yticklabels(toks, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Mean Attribution Score')
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'top_token_attributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: top_token_attributions.png")


def plot_marker_attribution(inside, outside, ratio):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Do Annotated Narrative Markers Drive DeBERTa\'s Predictions?\n'
                 '(validates narrative density as the model\'s core decision feature)',
                 fontsize=12, y=1.02)

    # Left: inside vs outside (pooled)
    ax = axes[0]
    all_in = [s for v in inside.values() for s in v]
    if all_in and outside:
        vp = ax.violinplot([all_in, outside], positions=[1, 2], widths=0.55,
                           showmedians=True, showextrema=False)
        for pc, col in zip(vp['bodies'], ['#f39c12', '#7f8c8d']):
            pc.set_facecolor(col); pc.set_alpha(0.75)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Inside\nMarker Spans', 'Outside\nSpans'])
        ax.set_ylabel('Attribution Score')
        ax.set_title('Attribution Inside vs. Outside Marker Spans')
        _, p = stats.mannwhitneyu(all_in, outside, alternative='greater')
        star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(0.97, 0.96, f'p={p:.4f} {star}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    # Right: ratio per marker type
    ax = axes[1]
    if ratio:
        mtypes = sorted(ratio.keys())
        means  = [np.mean(ratio[m]) for m in mtypes]
        sems   = [stats.sem(ratio[m]) if len(ratio[m]) > 1 else 0 for m in mtypes]
        colors_b = [MARKER_COLORS.get(m, '#aaa') for m in mtypes]
        y = np.arange(len(mtypes))
        ax.barh(y, means, xerr=sems, color=colors_b, alpha=0.85, capsize=4)
        ax.axvline(1.0, color='black', lw=1.2, ls='--', label='Baseline (1 = equal)')
        ax.set_yticks(y); ax.set_yticklabels(mtypes)
        ax.set_xlabel('Attribution Ratio (inside span / outside)\n>1 = marker tokens matter more')
        ax.set_title('Attribution Ratio by Marker Type')
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'marker_attribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: marker_attribution.png")


def plot_token_attr_bar(tokens, scores, label, filename):
    max_show = 50
    tokens = [_clean(t) for t in tokens[:max_show]]
    scores = scores[:max_show]
    norm   = scores / (scores.max() + 1e-9)
    cmap   = plt.get_cmap('RdYlGn')
    colors_bar = [cmap(float(s)) for s in norm]

    fig, ax = plt.subplots(figsize=(13, max(5, len(tokens)*0.23)))
    y = np.arange(len(tokens))
    ax.barh(y, norm, color=colors_bar)
    ax.set_yticks(y); ax.set_yticklabels(tokens, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Normalised Attribution')
    ax.set_title(f'Token Attribution — {label}', fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_attention_map(attn, tokens, label, filename):
    max_t = 40
    tokens = [_clean(t)[:12] for t in tokens[:max_t]]
    attn   = attn[:max_t, :max_t]
    n      = len(tokens)

    fig, ax = plt.subplots(figsize=(max(7, n*0.27), max(6, n*0.27)))
    im = ax.imshow(attn, cmap='Blues', aspect='auto')
    ax.set_xticks(range(n)); ax.set_xticklabels(tokens, rotation=60, ha='right', fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(tokens, fontsize=7)
    ax.set_title(f'Avg Attention (last 2 layers) — {label}', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_side_by_side_tp_fp(data, model, tokenizer, device, max_length=128):
    """Side-by-side attribution for a TP (correct Yes) and a FP (No predicted as Yes)."""
    model.eval()
    # Run predictions on a small batch to find TP/FP
    candidates = [d for d in data if d.get('markers')][:200]
    tp, fp = None, None

    for item in candidates:
        if tp and fp:
            break
        text  = get_text(item)
        true  = LABEL2ID[item['conspiracy']]
        enc   = tokenizer(text, truncation=True, max_length=max_length,
                          padding='max_length', return_tensors='pt')
        with torch.no_grad():
            logits, _ = model(enc['input_ids'].to(device),
                              enc['attention_mask'].to(device))
        pred = logits.argmax(-1).item()

        if pred == 1 and true == 1 and tp is None:
            tp = item
        if pred == 1 and true == 0 and fp is None:
            fp = item

    fig, axes = plt.subplots(1, 2, figsize=(22, 7))
    fig.suptitle('True Positive vs. False Positive Attribution\n'
                 '(why does the model mistake a No post for Yes?)',
                 fontsize=13, y=1.02)

    for ax, item, lbl, target in [
        (axes[0], tp, 'True Positive (Yes → predicted Yes)', 1),
        (axes[1], fp, 'False Positive (No → predicted Yes)',  1),
    ]:
        if item is None:
            ax.text(0.5, 0.5, f'No {lbl} example found in sample',
                    ha='center', va='center')
            ax.set_title(lbl); continue

        toks, scores = get_attribution(
            model, tokenizer, get_text(item), target, device, max_length
        )
        if toks is None:
            continue

        max_show = 40
        toks   = [_clean(t) for t in toks[:max_show]]
        scores = scores[:max_show]
        norm   = scores / (scores.max() + 1e-9)
        cmap   = plt.get_cmap('RdYlGn')
        colors_bar = [cmap(float(s)) for s in norm]

        y = np.arange(len(toks))
        ax.barh(y, norm, color=colors_bar)
        ax.set_yticks(y); ax.set_yticklabels(toks, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Normalised Attribution')
        ax.set_title(lbl, fontsize=10)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'tp_vs_fp_attribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: tp_vs_fp_attribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       default='data/train_rehydrated.jsonl')
    parser.add_argument('--model-name', default='microsoft/deberta-v3-base')
    parser.add_argument('--model-path', default=None,
                        help='Path to saved .pt checkpoint')
    parser.add_argument('--quick', action='store_true',
                        help='Use distilbert for fast local testing')
    parser.add_argument('--n-attr', type=int, default=200)
    parser.add_argument('--n-heat', type=int, default=3,
                        help='Number of attention heatmap examples per class')
    args = parser.parse_args()

    if args.quick:
        args.model_name = 'distilbert-base-uncased'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print("ANALYSIS 5: DEBERTA INTERPRETABILITY")
    print(f"{'='*60}")
    print(f"Model : {args.model_name}  |  Device: {device}")
    print(f"Figures → {FIG_DIR}/")

    print("\n[1/5] Loading data...")
    data     = load_data(args.data)
    yes_data = [d for d in data if d['conspiracy'] == 'Yes']
    no_data  = [d for d in data if d['conspiracy'] == 'No']
    print(f"  Yes: {len(yes_data)}  |  No: {len(no_data)}")

    print("\n[2/5] Model setup...")
    if args.model_path and Path(args.model_path).exists():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model     = load_checkpoint(args.model_path, args.model_name, device)
    else:
        if args.model_path:
            print(f"  [!] {args.model_path} not found — running quick train instead.")
        model, tokenizer = quick_train(data, args.model_name, device)
    model.eval()

    print("\n[3/5] Aggregate token attributions...")
    top_pos, top_neg = aggregate_top_tokens(
        data, model, tokenizer, device, n_samples=args.n_attr
    )
    print("  Top→Yes:", [t for t,_ in top_pos[:8]])
    print("  Top→No: ", [t for t,_ in top_neg[:8]])

    print("\n[4/5] Marker span attribution analysis...")
    inside, outside, ratio = analyse_marker_attributions(
        data, model, tokenizer, device, n_samples=min(args.n_attr, 120)
    )

    print("\n[5/5] Attention heatmaps & example attribution bars...")
    with_markers = [d for d in data if d.get('markers')]

    for i, item in enumerate(
        [d for d in with_markers if d['conspiracy'] == 'Yes'][:args.n_heat]
    ):
        t, s = get_attribution(model, tokenizer, get_text(item), 1, device)
        if t:
            plot_token_attr_bar(t, s, f'Believer (Yes) #{i+1}',
                                f'token_attr_yes_{i+1}.png')
        attn, toks = get_attention_map(model, tokenizer, get_text(item), device)
        if attn is not None:
            plot_attention_map(attn, toks, f'Believer (Yes) #{i+1}',
                               f'attention_yes_{i+1}.png')

    for i, item in enumerate(
        [d for d in with_markers if d['conspiracy'] == 'No'][:args.n_heat]
    ):
        t, s = get_attribution(model, tokenizer, get_text(item), 0, device)
        if t:
            plot_token_attr_bar(t, s, f'Non-Believer (No) #{i+1}',
                                f'token_attr_no_{i+1}.png')
        attn, toks = get_attention_map(model, tokenizer, get_text(item), device)
        if attn is not None:
            plot_attention_map(attn, toks, f'Non-Believer (No) #{i+1}',
                               f'attention_no_{i+1}.png')

    plot_side_by_side_tp_fp(data, model, tokenizer, device)

    print("\nGenerating summary figures...")
    plot_top_tokens(top_pos, top_neg)
    plot_marker_attribution(inside, outside, ratio)

    print(f"\n✓ All figures saved to {FIG_DIR}/")


if __name__ == '__main__':
    main()
