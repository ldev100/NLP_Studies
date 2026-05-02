# EPNI: Empirical Probabilistic Noise Injection for Clinical Noise Detection in Brazilian Portuguese

This repository contains the code, synthetic corpus, and trained models for the paper *"Empirical Probabilistic Noise Injection for Clinical Noise Detection in Brazilian Portuguese"*, submitted to BRACIS 2026.

## Overview

The EPNI framework transforms clean synthetic clinical narratives into realistic noisy text calibrated by real-world noise distributions. The framework operates in three stages:

1. **Clean Narrative Generation** — An LLM generates clean clinical anamneses in Brazilian Portuguese
2. **Empirical Noise Injection** — The EPNI module injects abbreviations and typographical errors based on an empirical noise taxonomy
3. **Downstream Evaluation** — Fine-tuned BERT models are evaluated on real clinical data for noise detection

## Repository Structure

```
epni-framework/
├── README.md
├── requirements.txt
│
├── scripts/
│   ├── injetar_ruido_ngram.py            # Stage 2: EPNI noise injection (n-gram matching)
│   ├── gerar_baseline_random.py          # Generates RANDOM baseline datasets
│   ├── fine_tuning_v3.py                 # Stage 3: Training and evaluation pipeline
│   ├── calcular_cobertura_vocabulario.py # Vocabulary coverage analysis
│   ├── verificar_anotacao.py             # Annotation quality verification
│   ├── corrigir_anotacao.py              # Taxonomy-assisted annotation correction
│
├── data/
│   ├── clean_synthetic_anamneses_gold_standart.json    # 1,150 clean synthetic anamneses
│   ├── clean_synthetic_anamneses_gold_standart_tokens.json  # Tokenized version (CLEAN baseline)
│   ├── template_mapeamento_top10.json    # Noise mapping: top 10 abbreviations + typos
│   ├── template_mapeamento_top15.json    # Noise mapping: top 15
│   └── template_mapeamento_top20.json    # Noise mapping: top 20
│
├── datasets_experimento_ruido/           # Generated training datasets
│   ├── dataset_top10_rate10.json         # EPNI: Top-10 vocabulary, 10% injection rate
│   ├── dataset_top10_rate25.json         # EPNI: Top-10, 25%
│   ├── dataset_top10_rate50.json         # EPNI: Top-10, 50%
│   ├── dataset_top15_rate10.json         # EPNI: Top-15, 10%
│   ├── dataset_top15_rate25.json         # EPNI: Top-15, 25%
│   ├── dataset_top15_rate50.json         # EPNI: Top-15, 50%
│   ├── dataset_top20_rate10.json         # EPNI: Top-20, 10%
│   ├── dataset_top20_rate25.json         # EPNI: Top-20, 25%
│   ├── dataset_top20_rate50.json         # EPNI: Top-20, 50%
│   ├── dataset_random_rate10.json        # RANDOM baseline, 10%
│   ├── dataset_random_rate25.json        # RANDOM baseline, 25%
│   └── dataset_random_rate50.json        # RANDOM baseline, 50%
│
├── modelos_finais/
│   ├── ClinNoiseBERT-base/              # Best BERTimbau fine-tuned model
│   └── ClinNoiseBERT-bio/               # Best BioBERTpt fine-tuned model
│
├── resultados_experimento/               # Experiment outputs
│   ├── metricas_completas.json           # All 78 runs with full metrics
│   ├── tabela_principal.csv              # Summary table
│   ├── tabela_latex.txt                  # LaTeX-formatted table
│   ├── grafico_heatmap.png               # ABBREV F1 heatmap
│   ├── grafico_barras.png                # ABBREV F1 bar chart
│   └── confusion_matrix_best.png         # Confusion matrix of best model
│
└── prompts/
    └── example_prompt.txt                # Representative generation prompt
```

## Scripts Description

**`injetar_ruido_ngram.py`**
Core EPNI framework. Reads clean anamneses and noise mapping templates, then injects abbreviations and typographical errors at configurable rates. Uses n-gram matching (4, 3, 2, 1 tokens) to correctly identify and replace multi-token clinical expressions (e.g., "ausculta pulmonar" → "AP", "bom estado geral" → "BEG"). Generates 9 EPNI datasets (3 vocabulary levels × 3 injection rates) plus the CLEAN baseline.

**`gerar_baseline_random.py`**
Generates 3 RANDOM baseline datasets by applying character-level perturbations (substitution, deletion, insertion, transposition) at matched intensity levels (10%, 25%, 50%) without domain-specific knowledge. These baselines isolate the contribution of empirical noise calibration.

**`fine_tuning_v3.py`**
Complete training and evaluation pipeline. Fine-tunes BERTimbau and BioBERTpt on each of the 13 training configurations with 3 random seeds (78 total runs). Evaluates on the real clinical test set and computes:
- Per-class F1 (CLEAN, ABBREV, TYPO)
- Macro F1 across noise classes
- Binary F1 (CLEAN vs. NOISE)
- Confusion matrix

### Annotation Quality Tools

**`calcular_cobertura_vocabulario.py`**
Calculates what percentage of noise tokens in the real test set are covered by the top-20 training vocabulary. Establishes the theoretical recall ceiling for the EPNI framework and identifies the most frequent uncovered noise patterns.

**`verificar_anotacao.py`**
Cross-references manual annotations against the complete noise taxonomy to identify potential classification errors. Reports consistency rate and flags tokens that may be mislabeled, serving as an automated second-pass quality check for single-annotator setups.

**`cruzar_typo_com_taxonomia.py`**
Specifically checks whether tokens labeled as TYPO in the test set appear in the abbreviation taxonomy, indicating they should be reclassified as ABBREV for consistency with the source taxonomy.

**`corrigir_anotacao.py`**
Applies corrections identified by the verification scripts. Separates corrections into automatic (unambiguous cases) and manual review (context-dependent cases). Creates backup before modifying the test set.

**`reclassificar_test_set.py`**
Reclassifies specific tokens from TYPO to ABBREV based on taxonomy cross-referencing. Used to maintain consistency between the test set annotations and the noise taxonomy from prior work.

## Data Description

### Synthetic Corpus
- **1,150 clinical anamneses** generated by Gemini 2.0 Flash
- Brazilian Portuguese, emergency department setting
- All clinical terms written in full form (no abbreviations)
- Validated against abbreviation checklist

### Noise Mapping Templates
Each template contains two categories:
- **Abbreviations**: clinical shorthand with full-form mappings (e.g., "BEG" → "bom estado geral")
- **Typographical errors**: common misspellings with correct forms (e.g., "hipogastro" → "hipogástrio")

Three vocabulary levels (Top-10, Top-15, Top-20) ordered by frequency in the source corpus.

### Training Datasets
Each dataset is a JSON array of tokenized anamneses with noise labels:
```json
[
  {
    "tokens": ["Pcte", "em", "BEG", ",", "abdome", "flácido"],
    "labels": ["CLEAN", "CLEAN", "ABBREV", "CLEAN", "CLEAN", "CLEAN"]
  }
]
```

### Test Set
The real clinical test set (500 anamneses, 37,394 tokens) is **not included** in this repository due to privacy regulations and institutional agreements. The test set distribution is 88.4% CLEAN, 10.8% ABBREV, and 0.8% TYPO.

## Trained Models

Two fine-tuned models are provided:

- **ClinNoiseBERT-base**: Fine-tuned from BERTimbau (`neuralmind/bert-base-portuguese-cased`) on TOP20-R50 configuration. Best ABBREV F1 = 0.826.
- **ClinNoiseBERT-bio**: Fine-tuned from BioBERTpt (`pucpr/biobertpt-all`) on TOP20-R25 configuration. Best ABBREV F1 = 0.809.

### Usage

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_path = "models/ClinNoiseBERT-base"
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Tokenize input
tokens = ["Pcte", "em", "BEG", "abdome", "flácido"]
inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")

# Predict
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)

# Labels: 0=CLEAN, 1=TYPO, 2=ABBREV
```

## Reproducing the Experiments

### Requirements

```bash
pip install torch transformers datasets evaluate seqeval scikit-learn pandas matplotlib seaborn
```

### Step-by-step

```bash
# 1. Generate EPNI datasets (requires noise mapping templates)
python scripts/injetar_ruido_ngram.py

# 2. Generate RANDOM baselines
python scripts/gerar_baseline_random.py

# 3. Run training and evaluation (requires test_set_real.json)
#    Estimated time: 13-20 hours on Apple M4 16GB
python scripts/fine_tuning_v3.py

# 4. Analyze vocabulary coverage (requires test_set_real.json)
python scripts/calcular_cobertura_vocabulario.py
```

## Key Results

| Configuration | Model | ABBREV F1 | Binary F1 |
|---|---|---|---|
| CLEAN | Both | 0.000 | 0.000 |
| RANDOM (best) | BioBERTpt | 0.000 | 0.331 |
| **TOP20-R50** | **BERTimbau** | **0.826** | **0.716** |
| TOP20-R25 | BioBERTpt | 0.809 | 0.687 |

## License

This project is released under the MIT License.

## Citation

```
[Citation will be added after acceptance]
```
