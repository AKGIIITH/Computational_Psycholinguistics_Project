#!/usr/bin/env python3
"""
Module D: Sentiment Arc & Discourse Coherence
===================================================
Analyzes the temporal and sequential dynamics of conspiracy text.
Moves beyond "bag of features" to measure narrative flow.

1. Discourse Coherence:
   Measures semantic similarity between sentence i and i+1 using
   MiniLM embeddings. 
   Hypothesis: Believers have lower sentence-to-sentence coherence
   (abrupt topic jumps, high surprisal) reflecting higher cognitive load.

2. Sentiment Arc (Narrative Trajectory):
   Splits posts into Beginning, Middle, and End.
   Hypothesis: Believers follow a "Revelation Arc" (start neutral/vague,
   drop into negative/alarming, end on a highly imperative/negative note).

Hardware: Safely utilizes GTX 1650 via lightweight sentence-transformers.
Output: figures/dynamics/ + results/coherence_arcs.csv
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

import torch
import nltk
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=1.1)

COLORS = {'Yes': '#e74c3c', 'No': '#3498db', "Can't tell": '#95a5a6'}
FIG_DIR = Path('figures/dynamics')
RES_DIR = Path('results')
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

# Ensure NLTK punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            lbl = d.get('conspiracy','').strip()
            if lbl in ('Yes', 'No'):
                rows.append({
                    'id': d.get('_id', ''),
                    'label': lbl,
                    'text': d.get('full_text', d.get('text', '')).strip()
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING DYNAMICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_dynamics(df, model, analyzer, device):
    """
    Computes coherence and sentiment arcs per post.
    """
    records = []
    
    # Process texts in batches for the transformer
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    ids = df['id'].tolist()
    
    for i in tqdm(range(len(texts)), desc='Processing narrative flow'):
        text = texts[i]
        label = labels[i]
        post_id = ids[i]
        
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        n_sent = len(sentences)
        
        # We need at least 3 sentences to compute a meaningful Beginning/Middle/End arc
        if n_sent < 3:
            continue
            
        # 1. Discourse Coherence (Semantic similarity between adjacent sentences)
        # Encode all sentences in this post
        with torch.no_grad():
            embeddings = model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
            
        # Compute cosine similarity for adjacent pairs: (s1, s2), (s2, s3), etc.
        similarities = []
        for j in range(n_sent - 1):
            sim = util.cos_sim(embeddings[j], embeddings[j+1]).item()
            similarities.append(sim)
            
        mean_coherence = np.mean(similarities)
        coherence_variance = np.var(similarities)
        
        # 2. Sentiment Arc (Beginning, Middle, End)
        sentiments = [analyzer.polarity_scores(s)['compound'] for s in sentences]
        
        # Divide into thirds (approximate)
        third = n_sent / 3.0
        part1 = sentiments[:int(np.ceil(third))]
        part3 = sentiments[int(np.floor(2*third)):]
        part2 = sentiments[int(np.ceil(third)):int(np.floor(2*third))]
        if not part2: # handle edge cases for very short arrays
            part2 = part1 
            
        records.append({
            'id': post_id,
            'label': label,
            'n_sentences': n_sent,
            'mean_coherence': mean_coherence,
            'coherence_variance': coherence_variance,
            'sent_mean': np.mean(sentiments),
            'sent_var': np.var(sentiments),
            'arc_begin': np.mean(part1),
            'arc_middle': np.mean(part2),
            'arc_end': np.mean(part3),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_discourse_coherence(df):
    """Violin plot comparing semantic coherence between classes."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    groups = [df[df['label'] == 'Yes']['mean_coherence'].values,
              df[df['label'] == 'No']['mean_coherence'].values]
              
    vp = ax.violinplot(groups, positions=[1, 2], widths=0.6,
                       showmedians=True, showextrema=False)
                       
    for pc, col in zip(vp['bodies'], [COLORS['Yes'], COLORS['No']]):
        pc.set_facecolor(col)
        pc.set_alpha(0.75)
        
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Believers (Yes)', 'Non-Believers (No)'])
    ax.set_ylabel('Mean Sentence-to-Sentence Cosine Similarity')
    ax.set_title('Discourse Coherence by Class\n'
                 'Lower coherence = abrupt topic jumps and higher cognitive load',
                 fontsize=12)
                 
    # Stats
    t_stat, p_val = stats.ttest_ind(groups[0], groups[1], equal_var=False)
    ax.text(0.95, 0.95, f'Welch t-test: p={p_val:.4f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'discourse_coherence.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'  Coherence Yes: {np.mean(groups[0]):.4f} ± {np.std(groups[0]):.4f}')
    print(f'  Coherence No:  {np.mean(groups[1]):.4f} ± {np.std(groups[1]):.4f}')
    print(f'  p-value:       {p_val:.6f}')


def plot_sentiment_arc(df):
    """Plots the narrative trajectory (Beginning -> Middle -> End)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    
    x = [1, 2, 3]
    x_labels = ['Beginning\n(Setup)', 'Middle\n(Complication)', 'End\n(Resolution/Imperative)']
    
    for lbl, color in [('Yes', COLORS['Yes']), ('No', COLORS['No'])]:
        sub = df[df['label'] == lbl]
        y = [sub['arc_begin'].mean(), sub['arc_middle'].mean(), sub['arc_end'].mean()]
        yerr = [sub['arc_begin'].sem(), sub['arc_middle'].sem(), sub['arc_end'].sem()]
        
        ax.errorbar(x, y, yerr=yerr, fmt='o-', color=color, lw=3, ms=8, capsize=5, 
                    label=f'{lbl} (n={len(sub)})')
                    
    ax.axhline(0, color='black', lw=1, ls='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_ylabel('Mean VADER Sentiment Compound Score\n(-1 = Negative, +1 = Positive)')
    ax.set_title('The Narrative Revelation Arc\n'
                 'How the emotional tone of a post evolves from start to finish',
                 fontsize=13)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'sentiment_arc.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_coherence_vs_sentiment_variance(df):
    """2D KDE plot showing the joint distribution of logic flow vs emotional flow."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    fig.suptitle('Cognitive Load vs. Emotional Volatility\n'
                 'Are believers emotionally volatile AND structurally disjointed?', 
                 fontsize=14, y=1.02)
    
    for ax, lbl in zip(axes, ['Yes', 'No']):
        sub = df[df['label'] == lbl]
        sns.kdeplot(data=sub, x='mean_coherence', y='sent_var', 
                    fill=True, cmap='Reds' if lbl=='Yes' else 'Blues', ax=ax, alpha=0.8)
        ax.set_title(f'True: {lbl}')
        ax.set_xlabel('Discourse Coherence (Semantic Similarity)')
        if lbl == 'Yes':
            ax.set_ylabel('Emotional Volatility (Sentiment Variance)')
        else:
            ax.set_ylabel('')
            
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'coherence_vs_volatility.png', dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/train_rehydrated.jsonl')
    args = parser.parse_args()

    print(f'\n{"="*60}')
    print('MODULE D: SENTIMENT ARC & DISCOURSE COHERENCE')
    print(f'{"="*60}')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device mapping: {device} (GTX 1650 Optimized)')

    print('\n[1/3] Loading models and data...')
    # Load fast sentence embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    # Load sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    df_raw = load_data(args.data)
    print(f'  Loaded {len(df_raw)} posts.')

    print('\n[2/3] Computing sentence trajectories (Arcs & Coherence)...')
    # This will filter out posts with < 3 sentences automatically
    df_dyn = compute_dynamics(df_raw, model, analyzer, device)
    print(f'  Successfully processed {len(df_dyn)} posts with >= 3 sentences.')
    
    df_dyn.to_csv(RES_DIR / 'coherence_arcs.csv', index=False)
    print(f'  Saved metrics to {RES_DIR}/coherence_arcs.csv')

    print('\n[3/3] Generating visual analytics...')
    plot_discourse_coherence(df_dyn)
    plot_sentiment_arc(df_dyn)
    plot_coherence_vs_sentiment_variance(df_dyn)
    
    print(f'\n✓ Figures saved to {FIG_DIR}/')
    print('='*60)

if __name__ == '__main__':
    main()