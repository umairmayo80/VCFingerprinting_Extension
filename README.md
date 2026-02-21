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

After running the notebook, find results in:

| File | Contents |
|------|----------|
| `results/tables/original_models.csv` | Paper models — all metrics |
| `results/tables/new_ml_models.csv` | New ML models |
| `results/tables/dl_models.csv` | DL models |
| `results/tables/comparison_all.csv` | All models combined |
| `results/tables/paper_vs_repro.csv` | Our results vs. paper's Table I |
| `results/figures/` | EDA plots + model comparison charts |

---