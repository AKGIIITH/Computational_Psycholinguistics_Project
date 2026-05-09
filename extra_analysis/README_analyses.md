# Conspiracy Belief Detection — Psycholinguistic Analysis Suite

Complete analysis suite for mid-project evaluation.
All scripts are standalone and independently runnable.

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Scripts at a Glance

| Script | Theme | Key Figures | GPU? |
|--------|-------|-------------|------|
| `analysis_01_surprisal_lexical.py` | Surprisal & Processing Load | surprisal distributions, positional trajectory, lexical frequency, readability | Yes (GPT-2) |
| `analysis_02_syntactic_pragmatic.py` | Syntax & Pragmatics | NP/VP, dep depth, passive voice, SVO networks, epistemic radar | No |
| `analysis_03_neuro_bridge.py` | Neuro-NLP Bridge | NER breakdown, factual vs. subjective scatter, Empath breakdown | No |
| `analysis_04_cosine.py` | Embedding Space | intra/inter similarity, SBERT vs TF-IDF, t-SNE, subreddit affinity | SBERT CPU ok |
| `analysis_05_interpretability.py` | DeBERTa Interpretability | token attributions, marker span analysis, attention heatmaps | Yes (optional) |
| `analysis_06_synthesis.py` | Master Synthesis | feature importance, correlation heatmap, **psycholinguistic radar**, pairwise scatter | No |

---

## Run Everything

```bash
# GTX 1650 (quick mode for interpretability):
./run_all_analyses.sh --data data/train_rehydrated.jsonl

# RTX 6000 with your saved checkpoint:
./run_all_analyses.sh \
  --data data/train_rehydrated.jsonl \
  --model-path subtask2/models/improved_seed2026.pt \
  --model-name microsoft/deberta-v3-large
```

---

## Script Details

### Analysis 1: Surprisal & Lexical Processing Load
**Theory:** Levy (2008) — surprisal(word) ∝ reading time. High-surprisal, low-frequency words → larger N400 ERP, longer fixation times.

```bash
python analysis_01_surprisal_lexical.py --data data/train_rehydrated.jsonl
```

**Figures produced:**
- `surprisal_distributions.png` — violin plots of mean surprisal, variance, clause boundary vs mid-clause
- `surprisal_positional.png` — surprisal trajectory from beginning to end of text
- `lexical_readability.png` — Zipf frequency, low-freq word ratio, FK grade, SMOG, TTR, MTLD
- `narrative_density_vs_readability.png` — scatter: marker count vs FK grade (does more narrative = harder text?)

**GTX 1650 note:** GPT-2 runs fine. Uses gpt2-medium (~1.5GB). Takes ~25 min on full dataset.

---

### Analysis 2: Syntactic Profiling & Epistemic Stance
**Theory:** Gibson DLT (1998) — deeper dependency trees = higher integration cost = longer reading time. Gricean Quality violations — booster/hedge asymmetry signals unverifiable claims.

```bash
python analysis_02_syntactic_pragmatic.py --data data/train_rehydrated.jsonl
```

**Figures produced:**
- `syntactic_profile.png` — NP/VP ratio, mean dep depth, passive voice, subordinate clause density
- `svo_networks.png` — NetworkX Subject-Verb-Object narrative motif graphs per class
- `pragmatic_markers.png` — hedge rate, certainty rate, pronoun profiles, vagueness attribution
- `epistemic_radar.png` — spider chart: Hedging / Certainty / 1st-singular / 1st-plural / 3rd-person / Vagueness

---

### Analysis 3: Neuroscience-to-NLP Bridge ⭐ KILLER SLIDE
**Theory:** Double dissociation from Nature paper (2025). Believers → vmPFC/dmPFC (value-based, emotional). Non-believers → Hippocampus/Precuneus (fact retrieval, semantic memory).

```bash
python analysis_03_neuro_bridge.py --data data/train_rehydrated.jsonl
```

**Figures produced:**
- `ner_breakdown.png` — NER density by entity type per class (Hippocampus proxy)
- `factual_vs_subjective_scatter.png` — **THE KILLER SLIDE**: every post plotted on Factual (NER) vs Subjective (Empath) axes, coloured by class, with centroids
- `neuro_feature_heatmap.png` — normalised mean of all neuro-proxy features per class
- `empath_breakdown.png` — Empath category scores per class (requires `pip install empath`)

**PPT tip:** Put the Nature paper brain scan on the left, your scatter plot on the right. The visual argument is immediate.

---

### Analysis 4: Cosine Similarity & Embedding Space
**Theory:** Shared narrative schema → tight intra-class embedding cluster. Hard examples (posts near the wrong centroid) = the model's failure zone.

```bash
python analysis_04_cosine.py --data data/train_rehydrated.jsonl

# Skip t-SNE (saves ~3 min):
python analysis_04_cosine.py --data data/train_rehydrated.jsonl --no-tsne

# Install SBERT first:
pip install sentence-transformers
```

**Figures produced:**
- `intra_inter_similarity.png` — violin: intra-Yes vs intra-No vs inter-class cosine similarity, for both SBERT and TF-IDF
- `sbert_vs_tfidf_separation.png` — centroid distance comparison: semantic vs lexical separation
- `sim_vs_marker_density.png` — scatter: cosine sim to Yes-centroid vs marker count (validates narrative density)
- `hard_examples.png` — text snippets of the most ambiguous posts in each class
- `tsne_embedding.png` — t-SNE projection of SBERT embeddings coloured by label
- `subreddit_affinity.png` — which subreddits cluster with believers vs non-believers

---

### Analysis 5: DeBERTa Interpretability
**Theory:** Which tokens and linguistic spans drive the model's predictions? Does it actually use narrative markers as expected by the narrative density thesis?

```bash
# GTX 1650 — quick mode (trains DistilBERT in ~20 min):
python analysis_05_interpretability.py --data data/train_rehydrated.jsonl --quick

# RTX 6000 — full DeBERTa-v3-large from checkpoint:
python analysis_05_interpretability.py \
  --data data/train_rehydrated.jsonl \
  --model-path subtask2/models/improved_seed2026.pt \
  --model-name microsoft/deberta-v3-large

# DeBERTa-v3-base (middle ground, no checkpoint needed):
python analysis_05_interpretability.py \
  --data data/train_rehydrated.jsonl \
  --model-name microsoft/deberta-v3-base
```

**Figures produced:**
- `top_token_attributions.png` — top 25 tokens driving Yes vs No predictions (bar charts)
- `marker_attribution.png` — inside vs outside marker span attribution, by marker type (validates narrative density)
- `token_attr_yes_N.png` — per-token attribution bar for selected Yes examples
- `token_attr_no_N.png` — per-token attribution bar for selected No examples
- `attention_yes_N.png` — attention heatmap for Yes examples
- `attention_no_N.png` — attention heatmap for No examples
- `tp_vs_fp_attribution.png` — side-by-side true positive vs false positive attribution (explains model errors)

---

### Analysis 6: Master Synthesis ⭐ MONEY SLIDE
**Combines all feature streams into one story.**

```bash
python analysis_06_synthesis.py --data data/train_rehydrated.jsonl
```

**Figures produced:**
- `feature_importance.png` — point-biserial correlation of all ~30 features with conspiracy label, sorted by magnitude
- `master_correlation_heatmap.png` — full feature × feature correlation matrix + label column
- `psycholinguistic_radar.png` — **THE MONEY SLIDE**: single radar chart comparing Believers vs Non-Believers across 10 psycholinguistic dimensions simultaneously
- `pairwise_scatter.png` — scatter matrix for top 6 discriminating features

---

## Estimated Runtime (GTX 1650, full dataset ~4800 posts)

| Script | Time |
|--------|------|
| Analysis 1 (GPT-2 surprisal) | ~25 min |
| Analysis 2 (spaCy syntactic) | ~12 min |
| Analysis 3 (NER + Empath) | ~8 min |
| Analysis 4 (SBERT + t-SNE) | ~20 min |
| Analysis 5 (quick DistilBERT) | ~25 min |
| Analysis 6 (synthesis) | ~8 min |
| **Total** | **~1.5 hours** |

**RTX 6000 total:** ~30 min (full DeBERTa-v3-large).

---

## Figure → PPT Slide Mapping

| Slide | Figure(s) |
|-------|-----------|
| Dataset Overview | dataset stats (manual) |
| Surprisal Analysis | `01/surprisal_distributions.png`, `01/surprisal_positional.png` |
| Lexical Processing Load | `01/lexical_readability.png` |
| Syntactic Profiling | `02/syntactic_profile.png` |
| Narrative Motifs | `02/svo_networks.png` |
| Pragmatics / Gricean | `02/pragmatic_markers.png`, `02/epistemic_radar.png` |
| **Neuro Bridge (KILLER SLIDE)** | `03/factual_vs_subjective_scatter.png` + Nature paper brain image |
| Empath Breakdown | `03/empath_breakdown.png` |
| Cosine / Embedding Space | `04/intra_inter_similarity.png`, `04/sbert_vs_tfidf_separation.png`, `04/tsne_embedding.png` |
| Subreddit Affinity | `04/subreddit_affinity.png` |
| Hard Examples | `04/hard_examples.png` |
| Model Interpretability | `05/top_token_attributions.png`, `05/marker_attribution.png`, `05/tp_vs_fp_attribution.png` |
| Feature Importance | `06/feature_importance.png` |
| **Summary (MONEY SLIDE)** | `06/psycholinguistic_radar.png` |
| Conclusions | manual |

---

## Theoretical Connections (for PPT narrative)

- **Surprisal + N400:** High surprisal conspiracy words → predict larger N400 (semantic integration cost) — connects to ERP literature in psycholinguistics
- **DLT + Passive voice:** Deep trees + passive voice → longer reading time, implies deliberate obscuring of causal agency
- **Gricean Quality:** Boosters compensate for epistemically unverifiable claims; hedges provide deniability — both signals fire simultaneously in believers
- **RSA (Bayesian pragmatics):** Conspiracy discourse is only pragmatically rational under a very different prior P(world) — your surprisal numbers operationalise this
- **Nature paper bridge:** vmPFC/dmPFC → value/belief = your Empath moral-emotional score; Hippocampus/Precuneus → fact retrieval = your NER factual grounding score
