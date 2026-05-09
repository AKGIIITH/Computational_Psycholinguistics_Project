#!/usr/bin/env python3
"""
Analysis 4: Cosine Similarity & Embedding Space
================================================
Uses Sentence-BERT to embed all posts and analyses the semantic geometry
of the conspiracy vs. non-conspiracy text space.

Analyses:
  1. Class centroid cosine distance — how far apart are the two class means?
  2. Intra-class vs inter-class similarity distributions (violin)
  3. TF-IDF cosine similarity vs. SBERT comparison — semantic vs. lexical sep.
  4. Hard examples: No-posts closest to Yes-centroid (likely model FPs)
                    Yes-posts closest to No-centroid (likely model FNs)
  5. Scatter: cosine similarity to Yes-centroid vs. narrative density marker count
  6. t-SNE / UMAP projection of the embedding space, coloured by label
  7. Subreddit-level centroids — which subreddits cluster with believers?

Theory grounding:
  - Tight intra-class cluster → shared narrative schema (conspiracy as genre)
  - Hard examples = semantically ambiguous posts → model confusion region
  - SBERT >> TF-IDF separation confirms semantic (not surface) difference

Output: figures/04_cosine/

Usage:
  python analysis_04_cosine.py --data data/train_rehydrated.jsonl
  python analysis_04_cosine.py --data data/train_rehydrated.jsonl --umap   # if umap-learn installed
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
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from tqdm import tqdm

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=1.1)
COLORS = {'Yes': '#e74c3c', 'No': '#3498db', 'cant': '#95a5a6'}
FIG_DIR = Path('figures/04_cosine')
FIG_DIR.mkdir(parents=True, exist_ok=True)


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


def get_marker_count(item):
    return len(item.get('markers', []))


# ─────────────────────────────────────────────────────────────────────────────
# SBERT EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

def embed_sbert(texts, model_name='all-MiniLM-L6-v2', batch_size=64):
    """
    Embed texts using SentenceTransformer.
    Falls back to TF-IDF + SVD if sentence-transformers not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
        print(f"  Loading SentenceTransformer ({model_name})...")
        model = SentenceTransformer(model_name)
        print(f"  Encoding {len(texts)} texts (batch_size={batch_size})...")
        embeddings = model.encode(
            texts, batch_size=batch_size, show_progress_bar=True,
            normalize_embeddings=True,   # L2-normalised → cosine sim = dot product
            convert_to_numpy=True
        )
        return embeddings, 'SBERT'
    except ImportError:
        print("[WARNING] sentence-transformers not installed.")
        print("  Install: pip install sentence-transformers")
        print("  Falling back to TF-IDF + TruncatedSVD (300d)...")
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize
        vec = TfidfVectorizer(max_features=20000, sublinear_tf=True)
        X   = vec.fit_transform(texts)
        svd = TruncatedSVD(n_components=300, random_state=42)
        E   = normalize(svd.fit_transform(X), norm='l2')
        return E, 'TF-IDF+SVD (fallback)'


# ─────────────────────────────────────────────────────────────────────────────
# TFIDF BASELINE EMBEDDINGS (for comparison)
# ─────────────────────────────────────────────────────────────────────────────

def embed_tfidf(texts, n_components=300):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    vec = TfidfVectorizer(max_features=15000, sublinear_tf=True,
                          min_df=2, ngram_range=(1, 2))
    X   = vec.fit_transform(texts)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    E   = normalize(svd.fit_transform(X), norm='l2')
    return E


# ─────────────────────────────────────────────────────────────────────────────
# COSINE ANALYSES
# ─────────────────────────────────────────────────────────────────────────────

def class_centroid_distance(yes_emb, no_emb):
    """Cosine distance between class centroids (L2-normalised embeddings)."""
    yes_centroid = yes_emb.mean(axis=0)
    no_centroid  = no_emb.mean(axis=0)
    # L2-normalise centroids
    yes_centroid /= np.linalg.norm(yes_centroid) + 1e-9
    no_centroid  /= np.linalg.norm(no_centroid)  + 1e-9
    dist = float(cosine(yes_centroid, no_centroid))
    sim  = 1.0 - dist
    print(f"  Centroid cosine similarity: {sim:.4f}  (distance: {dist:.4f})")
    return yes_centroid, no_centroid, sim, dist


def intra_inter_similarity(yes_emb, no_emb, sample_n=300, seed=42):
    """
    Sample pairwise cosine similarities:
      - intra-Yes, intra-No, inter (Yes↔No)
    Returns dict of lists.
    """
    rng = np.random.default_rng(seed)

    def _sample_pairs(A, B=None, n=sample_n):
        if B is None:   # intra-class
            idx  = rng.choice(len(A), size=min(n*2, len(A)), replace=False)
            half = len(idx) // 2
            sims = (A[idx[:half]] * A[idx[half:]]).sum(axis=1)   # dot = cosine if L2-normed
        else:           # inter-class
            ia = rng.choice(len(A), size=min(n, len(A)), replace=False)
            ib = rng.choice(len(B), size=min(n, len(B)), replace=False)
            sims = (A[ia] * B[ib]).sum(axis=1)
        return sims.tolist()

    return {
        'intra_yes': _sample_pairs(yes_emb),
        'intra_no':  _sample_pairs(no_emb),
        'inter':     _sample_pairs(yes_emb, no_emb),
    }


def hard_examples(all_data, all_emb, yes_centroid, no_centroid, top_n=10):
    """
    Find:
      - No-posts closest to Yes-centroid (false-positive candidates)
      - Yes-posts closest to No-centroid (false-negative candidates)
    """
    labels = [d['conspiracy'] for d in all_data]
    texts  = [get_text(d)      for d in all_data]

    no_idx  = [i for i, l in enumerate(labels) if l == 'No']
    yes_idx = [i for i, l in enumerate(labels) if l == 'Yes']

    # No posts → similarity to Yes centroid
    no_emb_arr   = all_emb[no_idx]
    sim_to_yes   = no_emb_arr @ yes_centroid   # dot product (L2-normalised)
    top_fp_local = np.argsort(sim_to_yes)[-top_n:][::-1]
    fp_examples  = [{'text': texts[no_idx[i]][:200],
                     'sim':  float(sim_to_yes[i]),
                     'label': 'No → ≈Yes (likely FP)'}
                    for i in top_fp_local]

    # Yes posts → similarity to No centroid
    yes_emb_arr  = all_emb[yes_idx]
    sim_to_no    = yes_emb_arr @ no_centroid
    top_fn_local = np.argsort(sim_to_no)[-top_n:][::-1]
    fn_examples  = [{'text': texts[yes_idx[i]][:200],
                     'sim':  float(sim_to_no[i]),
                     'label': 'Yes → ≈No (likely FN)'}
                    for i in top_fn_local]

    return fp_examples, fn_examples


def similarity_vs_narrative_density(all_data, all_emb, yes_centroid):
    """Scatter: cosine sim to Yes-centroid vs. marker count."""
    labels  = [d['conspiracy']    for d in all_data]
    markers = [get_marker_count(d) for d in all_data]
    sims    = (all_emb @ yes_centroid).tolist()
    return labels, markers, sims


# ─────────────────────────────────────────────────────────────────────────────
# SUBREDDIT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def subreddit_centroids(all_data, all_emb, yes_centroid, no_centroid,
                         min_posts=15, top_n=20):
    """
    Per-subreddit mean embedding, then compute similarity to each class centroid.
    """
    sub_idx = defaultdict(list)
    for i, d in enumerate(all_data):
        sr = d.get('subreddit', 'unknown')
        sub_idx[sr].append(i)

    rows = []
    for sr, idxs in sub_idx.items():
        if len(idxs) < min_posts:
            continue
        emb  = all_emb[idxs].mean(axis=0)
        emb /= np.linalg.norm(emb) + 1e-9
        sim_yes = float(emb @ yes_centroid)
        sim_no  = float(emb @ no_centroid)
        # Conspiracy affinity = sim_yes - sim_no
        rows.append({'subreddit': sr, 'sim_yes': sim_yes,
                     'sim_no': sim_no,
                     'affinity': sim_yes - sim_no,
                     'n': len(idxs),
                     'pct_yes': sum(
                         1 for i in idxs
                         if all_data[i]['conspiracy'] == 'Yes'
                     ) / len(idxs)})

    rows.sort(key=lambda r: r['affinity'], reverse=True)
    return rows[:top_n], rows[-top_n:][::-1]


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_intra_inter(sims_sbert, sims_tfidf):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Intra- vs. Inter-Class Cosine Similarity\n'
                 '(tight intra-class = shared narrative schema)',
                 fontsize=13, y=1.02)

    for ax, sims, title in [
        (axes[0], sims_sbert, 'Semantic (SBERT)'),
        (axes[1], sims_tfidf, 'Lexical (TF-IDF + SVD 300d)'),
    ]:
        groups = [sims['intra_yes'], sims['intra_no'], sims['inter']]
        labels_v = ['Intra\nBeliever', 'Intra\nNon-Believer', 'Inter\n(cross-class)']
        clrs    = [COLORS['Yes'], COLORS['No'], '#95a5a6']

        vp = ax.violinplot(groups, positions=[1, 2, 3], widths=0.55,
                           showmedians=True, showextrema=False)
        for pc, col in zip(vp['bodies'], clrs):
            pc.set_facecolor(col)
            pc.set_alpha(0.75)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(labels_v)
        ax.set_ylabel('Cosine Similarity')
        ax.set_title(title)

        # Annotate medians
        for i, g in enumerate(groups, 1):
            ax.text(i, np.median(g) + 0.005,
                    f'{np.median(g):.3f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'intra_inter_similarity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: intra_inter_similarity.png")


def plot_sbert_vs_tfidf_centroid(dist_sbert, dist_tfidf):
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ['SBERT\n(semantic)', 'TF-IDF+SVD\n(lexical)']
    dists   = [dist_sbert, dist_tfidf]
    bars = ax.bar(methods, dists, color=['#2ecc71', '#e67e22'], alpha=0.85, width=0.4)
    for bar, val in zip(bars, dists):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Centroid Cosine Distance\n(higher = more separable)')
    ax.set_title('Semantic vs. Lexical Class Separation\n'
                 '(SBERT captures meaning, TF-IDF captures surface words)')
    ax.set_ylim(0, max(dists) * 1.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'sbert_vs_tfidf_separation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: sbert_vs_tfidf_separation.png")


def plot_sim_vs_marker_density(labels, markers, sims):
    fig, ax = plt.subplots(figsize=(11, 6))
    for lbl, color in [('Yes', COLORS['Yes']), ('No', COLORS['No'])]:
        idx = [i for i, l in enumerate(labels) if l == lbl]
        xs  = [markers[i] for i in idx]
        ys  = [sims[i]    for i in idx]
        ax.scatter(xs, ys, alpha=0.25, color=color, s=20,
                   edgecolors='none', label=f'{lbl} posts')
        if len(xs) > 5:
            m, b, r, p, _ = stats.linregress(xs, ys)
            xl = np.array([min(xs), max(xs)])
            ax.plot(xl, m*xl + b, '--', color=color, lw=2,
                    label=f'{lbl} trend r={r:.3f}')
    ax.set_xlabel('Narrative Density (marker count per post)')
    ax.set_ylabel('Cosine Similarity to Conspiracy-Believer Centroid')
    ax.set_title('Does Higher Narrative Density → Closer to Believer Embedding?\n'
                 '(validates narrative density as the core separating feature)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'sim_vs_marker_density.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: sim_vs_marker_density.png")


def plot_hard_examples(fp_examples, fn_examples):
    fig, axes = plt.subplots(1, 2, figsize=(20, max(6, len(fp_examples)*0.55 + 2)))
    fig.suptitle('Hard Examples: Semantically Ambiguous Posts\n'
                 '(posts closest to the "wrong" class centroid — likely model errors)',
                 fontsize=13, y=1.02)

    for ax, examples, title, color in [
        (axes[0], fp_examples, 'False-Positive Candidates\n(No posts near Yes centroid)',
         COLORS['Yes']),
        (axes[1], fn_examples, 'False-Negative Candidates\n(Yes posts near No centroid)',
         COLORS['No']),
    ]:
        ax.set_title(title, fontsize=11)
        ax.axis('off')
        y = 0.97
        for ex in examples:
            snippet = ex['text'][:130].replace('\n', ' ') + '…'
            ax.text(0.01, y, f"[sim={ex['sim']:.3f}]", transform=ax.transAxes,
                    fontsize=7, color=color, fontweight='bold', va='top')
            ax.text(0.13, y, snippet, transform=ax.transAxes,
                    fontsize=7, va='top', wrap=True,
                    bbox=dict(boxstyle='round,pad=0.2', fc='#f8f8f8', alpha=0.5))
            y -= 0.1

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'hard_examples.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: hard_examples.png")


def plot_tsne(all_data, all_emb, perplexity=40):
    print("  Running t-SNE (may take 1-3 min)...")
    tsne_kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init='pca',
        learning_rate='auto',
    )
    # scikit-learn renamed this arg in newer versions.
    try:
        tsne = TSNE(**tsne_kwargs, max_iter=1000)
    except TypeError:
        tsne = TSNE(**tsne_kwargs, n_iter=1000)
    coords = tsne.fit_transform(all_emb)

    labels = [d['conspiracy'] for d in all_data]
    subs   = [d.get('subreddit', '') for d in all_data]

    fig, ax = plt.subplots(figsize=(12, 9))
    for lbl, color in [('Yes', COLORS['Yes']), ('No', COLORS['No'])]:
        idx = [i for i, l in enumerate(labels) if l == lbl]
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   alpha=0.35, s=15, color=color,
                   edgecolors='none', label=f'{lbl} ({len(idx)} posts)')

    ax.set_title(f't-SNE Projection of SBERT Embeddings\n'
                 f'(perplexity={perplexity})', fontsize=13)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.legend(markerscale=3)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'tsne_embedding.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: tsne_embedding.png")


def plot_subreddit_affinity(top_conspiracy, top_normal):
    fig, ax = plt.subplots(figsize=(13, 8))

    all_rows = top_conspiracy + top_normal
    names    = [r['subreddit'] for r in all_rows]
    affinities = [r['affinity'] for r in all_rows]
    pct_yes  = [r['pct_yes']   for r in all_rows]
    colors   = [COLORS['Yes'] if a > 0 else COLORS['No'] for a in affinities]

    y = np.arange(len(names))
    bars = ax.barh(y, affinities, color=colors, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Conspiracy Affinity Score\n'
                  '(cosine sim to Yes-centroid − sim to No-centroid)')
    ax.set_title('Subreddit Conspiracy Affinity\n'
                 '(derived from SBERT embedding proximity to each class centroid)',
                 fontsize=12)

    # Annotate with % Yes label
    for i, (bar, row) in enumerate(zip(bars, all_rows)):
        ax.text(bar.get_width() + 0.001 if bar.get_width() >= 0 else bar.get_width() - 0.001,
                i, f" {row['pct_yes']*100:.0f}% Yes  n={row['n']}",
                va='center', fontsize=7,
                ha='left' if bar.get_width() >= 0 else 'right')

    patches = [
        mpatches.Patch(color=COLORS['Yes'], label='High conspiracy affinity'),
        mpatches.Patch(color=COLORS['No'],  label='Low conspiracy affinity'),
    ]
    ax.legend(handles=patches, loc='lower right')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'subreddit_affinity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: subreddit_affinity.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/train_rehydrated.jsonl')
    parser.add_argument('--sbert-model', default='all-MiniLM-L6-v2',
                        help='SentenceTransformer model name')
    parser.add_argument('--no-tsne', action='store_true',
                        help='Skip t-SNE (saves ~3 min on large data)')
    parser.add_argument('--sample', type=int, default=None,
                        help='Subsample N examples for speed (default: all)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("ANALYSIS 4: COSINE SIMILARITY & EMBEDDING SPACE")
    print(f"{'='*60}")
    print(f"Figures → {FIG_DIR}/")

    data = load_data(args.data)
    if args.sample:
        import random
        random.seed(42)
        data = random.sample(data, min(args.sample, len(data)))
        print(f"  [Subsampled to {len(data)} posts]")

    yes_data = [d for d in data if d['conspiracy'] == 'Yes']
    no_data  = [d for d in data if d['conspiracy'] == 'No']
    print(f"  Yes: {len(yes_data)}  |  No: {len(no_data)}")

    all_texts  = [get_text(d) for d in data]
    yes_texts  = [get_text(d) for d in yes_data]
    no_texts   = [get_text(d) for d in no_data]

    # ── 1. SBERT embeddings ──────────────────────────────────────────────────
    print("\n[1/5] Computing SBERT embeddings...")
    all_emb, embed_name = embed_sbert(all_texts, args.sbert_model)
    yes_emb = all_emb[[i for i, d in enumerate(data) if d['conspiracy'] == 'Yes']]
    no_emb  = all_emb[[i for i, d in enumerate(data) if d['conspiracy'] == 'No']]

    # ── 2. Class centroid distances ──────────────────────────────────────────
    print("\n[2/5] Centroid distances...")
    yes_centroid, no_centroid, sim_sbert, dist_sbert = class_centroid_distance(
        yes_emb, no_emb
    )

    # TF-IDF baseline for comparison
    print("  Computing TF-IDF baseline...")
    tfidf_emb     = embed_tfidf(all_texts)
    yes_tfidf     = tfidf_emb[[i for i, d in enumerate(data) if d['conspiracy'] == 'Yes']]
    no_tfidf      = tfidf_emb[[i for i, d in enumerate(data) if d['conspiracy'] == 'No']]
    yc_tf = yes_tfidf.mean(axis=0); yc_tf /= np.linalg.norm(yc_tf) + 1e-9
    nc_tf = no_tfidf.mean(axis=0);  nc_tf  /= np.linalg.norm(nc_tf) + 1e-9
    dist_tfidf = float(cosine(yc_tf, nc_tf))
    print(f"  TF-IDF centroid cosine distance: {dist_tfidf:.4f}")

    # ── 3. Intra/inter similarity ────────────────────────────────────────────
    print("\n[3/5] Intra/inter-class similarity sampling...")
    sims_sbert = intra_inter_similarity(yes_emb, no_emb)
    sims_tfidf = intra_inter_similarity(yes_tfidf, no_tfidf)

    # ── 4. Hard examples ─────────────────────────────────────────────────────
    print("\n[4/5] Hard examples...")
    fp_ex, fn_ex = hard_examples(data, all_emb, yes_centroid, no_centroid)
    print(f"  Top FP candidate: '{fp_ex[0]['text'][:80]}...'")
    print(f"  Top FN candidate: '{fn_ex[0]['text'][:80]}...'")

    # ── 5. Similarity vs. marker density ─────────────────────────────────────
    labels, markers, sims = similarity_vs_narrative_density(
        data, all_emb, yes_centroid
    )

    # ── 6. Subreddit affinity ────────────────────────────────────────────────
    top_con, top_norm = subreddit_centroids(data, all_emb, yes_centroid, no_centroid)

    # ── Plotting ─────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_intra_inter(sims_sbert, sims_tfidf)
    plot_sbert_vs_tfidf_centroid(dist_sbert, dist_tfidf)
    plot_sim_vs_marker_density(labels, markers, sims)
    plot_hard_examples(fp_ex, fn_ex)
    if not args.no_tsne:
        plot_tsne(data, all_emb)
    plot_subreddit_affinity(top_con, top_norm)

    # Print summary
    print(f"\n{'─'*50}")
    print("COSINE SIMILARITY SUMMARY")
    print(f"  Embedding method:           {embed_name}")
    print(f"  SBERT centroid dist:        {dist_sbert:.4f}  (sim={sim_sbert:.4f})")
    print(f"  TF-IDF centroid dist:       {dist_tfidf:.4f}")
    print(f"  SBERT intra-Yes median sim: {np.median(sims_sbert['intra_yes']):.4f}")
    print(f"  SBERT intra-No  median sim: {np.median(sims_sbert['intra_no']):.4f}")
    print(f"  SBERT inter     median sim: {np.median(sims_sbert['inter']):.4f}")
    print(f"  Figures saved to: {FIG_DIR}/")


if __name__ == '__main__':
    main()
