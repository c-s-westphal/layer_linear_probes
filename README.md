# Linear Probe PCA Experiment

Trains linear probes on PCA-reduced activations from GPT-2 Small to predict linguistic features across all 12 layers.

## Dataset Summary

### Plurality Task (1,000 examples)
- **500 unique singular examples**: Diverse nouns (animals, professions, objects, nature) with singular verb agreement
- **500 unique plural examples**: Matching plural forms with plural verb agreement
- **Coverage**: 50 animals, 100 professions, 200 objects, 150 nature/places

### Part-of-Speech Task (800 examples)
- **200 unique nouns**: Abstract (60), animals (30), objects (50), nature (30), people (30)
- **200 unique verbs**: Motion (40), communication (40), cognitive (40), physical action (40), other (40)
- **200 unique adjectives**: Colors (20), size (20), temperature (20), speed (20), quality (40), emotions (40), physical (40)
- **200 unique adverbs**: Manner (80), time (40), place/direction (40), degree (40)

All examples are unique with diverse vocabulary, sentence structures, and target word positions.

## Files Created

```
layer_linear_probes/
├── linear_probe_pca_experiment.py  # Main experiment script
├── pos_dataset_generator.py        # Comprehensive POS dataset (800 examples)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore patterns
└── src/
    ├── __init__.py
    └── model.py                    # ModelLoader class for GPT-2
```

## Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install manually
pip install transformers torch scikit-learn matplotlib numpy scipy pandas tqdm sae-lens transformer-lens
```

## Usage

### Basic Usage
```bash
python linear_probe_pca_experiment.py
```

### With Custom Parameters
```bash
python linear_probe_pca_experiment.py \
    --output_dir outputs/my_experiment \
    --n_components 10 \
    --n_runs 3 \
    --seed 42
```

### Parameters

- `--output_dir`: Output directory for results (default: `outputs/linear_probe_pca`)
- `--n_components`: Number of PCA components (default: 10)
- `--n_runs`: Number of probe training runs for confidence intervals (default: 3)
- `--seed`: Random seed (default: 42)

## Output Structure

```
outputs/linear_probe_pca/
├── experiment.log              # Detailed execution log
├── raw_results.csv            # All measurements (layer, task, run, MI, accuracy, F1)
└── plots/
    ├── plurality_mutual_information.png
    ├── plurality_accuracy.png
    ├── pos_mutual_information.png
    └── pos_accuracy.png
```

## Experiment Pipeline

For each of 11 layers (1-11, skipping layer 0 input embeddings):

1. **Extract Activations** (768-dim)
   - Process all examples through GPT-2
   - Extract activation at last token of target word
   - Handle multi-token words correctly

2. **Apply PCA**
   - Fit PCA on layer activations
   - Reduce to top 10 components
   - Log explained variance (per-component and cumulative)

3. **Train Probes** (3 runs per layer)
   - Train LogisticRegression on PCA-reduced activations
   - No train/test split (train and evaluate on all data)
   - Calculate metrics: Mutual Information, Accuracy, F1 Score

4. **Aggregate Results**
   - Compute mean and 95% confidence intervals across 3 runs
   - Generate bar plots with error bars

## Metrics

- **Mutual Information**: `I(predictions; true_labels)` using `sklearn.metrics.mutual_info_score`
- **Classification Accuracy**: Proportion of correct predictions
- **F1 Score**: Macro-averaged F1 (handles class imbalance in multi-class)

## Expected Runtime

- **GPU**: ~10-20 minutes
- **CPU**: ~30-60 minutes (depending on hardware)

## Expected Results Pattern

Based on probing literature, typical findings:

- **Early layers (1-3)**: Low accuracy/MI (surface features only)
- **Middle layers (4-8)**: Peak accuracy/MI (syntactic and semantic features)
- **Late layers (9-11)**: Variable (task-specific, may decrease)

Different tasks peak at different layers:
- **Plurality**: Often peaks in middle layers (syntactic agreement)
- **POS**: May peak earlier (part-of-speech is more syntactic)

## Notes

- The script automatically uses GPU if available (CUDA)
- All datasets contain unique examples (no repetition)
- Confidence intervals use Student's t-distribution (`scipy.stats.t.interval`)
- Plot style matches the provided `random_sampling.py` reference
- Logging is comprehensive - check `experiment.log` for detailed progress

## Citation

If you use this code, please cite:

```bibtex
@misc{linear_probe_pca,
  title={Linear Probe PCA Experiment on GPT-2 Small},
  author={Your Name},
  year={2025}
}
```
