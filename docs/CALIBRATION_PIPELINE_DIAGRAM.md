# Pipeline Calibrazione e Cost-Sensitive Decision - Diagramma di Flusso

## 📊 Architettura Completa

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATASET COMPLETO (ADNI)                          │
│                  ↓ processing_features_cv_with_calibration()        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │   Cross-Validation Split  │
                    │      (5 Folds)            │
                    └─────────────┬─────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
            ▼                     ▼                     ▼
    ┌──────────────┐      ┌──────────────┐    ┌──────────────┐
    │   TRAINING   │      │ CALIBRATION  │    │     TEST     │
    │     60%      │      │     20%      │    │     20%      │
    │ (X_train,    │      │ (X_calib,    │    │ (X_test,     │
    │  y_train)    │      │  y_calib)    │    │  y_test)     │
    └──────┬───────┘      └──────┬───────┘    └──────┬───────┘
           │                     │                     │
           │                     │                     │
           ▼                     │                     │
    ┌──────────────────────┐    │                     │
    │   MODEL TRAINING     │    │                     │
    │  IRBoostSH Ensemble  │    │                     │
    │  - Clinical RF       │    │                     │
    │  - Image CNN         │    │                     │
    └──────────┬───────────┘    │                     │
               │                 │                     │
               │    Predict      │                     │
               │◄────────────────┘                     │
               │                                       │
               ▼                                       │
    ┌──────────────────────┐                          │
    │  UNCALIBRATED PROBS  │                          │
    │   p_calib_uncal      │                          │
    │  [n_calib × 3]       │                          │
    └──────────┬───────────┘                          │
               │                                       │
               ▼                                       │
    ┌──────────────────────────────────┐              │
    │   ISOTONIC REGRESSION FITTING    │              │
    │   (One regressor per class)      │              │
    │   - CN calibrator                │              │
    │   - MCI calibrator               │              │
    │   - AD calibrator                │              │
    └──────────┬───────────────────────┘              │
               │                                       │
               │              Model Predict            │
               │◄──────────────────────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  UNCALIBRATED PROBS  │
    │   p_test_uncal       │
    │   [n_test × 3]       │
    └──────────┬───────────┘
               │
               │ Apply IR Transform
               ▼
    ┌──────────────────────┐
    │  CALIBRATED PROBS    │
    │   p_test_cal         │
    │   [n_test × 3]       │
    │  ✓ ECE Improved      │
    │  ✓ Well-calibrated   │
    └──────────┬───────────┘
               │
               └────────┬────────────────────────────┐
                        │                            │
               ┌────────▼────────┐        ┌──────────▼──────────┐
               │  STANDARD       │        │  COST-SENSITIVE     │
               │  DECISION       │        │  BAYESIAN DECISION  │
               │  (argmax)       │        │  Rule               │
               │                 │        │                     │
               │  ŷ = argmax p_i │        │  ŷ = argmin Σ C·p_i │
               └────────┬────────┘        └──────────┬──────────┘
                        │                            │
                        │                            │
               ┌────────▼────────┐        ┌──────────▼──────────┐
               │   y_pred_std    │        │  y_pred_cost_sens   │
               │                 │        │                     │
               │  Max accuracy   │        │  Min clinical cost  │
               │  Higher cost    │        │  Lower cost ✓       │
               └────────┬────────┘        └──────────┬──────────┘
                        │                            │
                        └────────┬───────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │    EVALUATION          │
                    │  - Accuracy            │
                    │  - Precision/Recall/F1 │
                    │  - Confusion Matrix    │
                    │  - Clinical Cost       │
                    │  - ECE/MCE/Brier       │
                    └────────┬───────────────┘
                             │
                             ▼
                    ┌─────────────────────────┐
                    │  RESULTS & ANALYSIS     │
                    │  - Metrics CSV          │
                    │  - Visualizations       │
                    │  - Summary Report       │
                    └─────────────────────────┘
```

## 🔄 Flusso Dettagliato per Componente

### 1. Isotonic Regression Calibration

```
INPUT: p_uncal [n × 3], y_true [n]
│
├─ For each class c ∈ {CN, MCI, AD}:
│  │
│  ├─ Extract p_c = p_uncal[:, c]
│  │
│  ├─ Binary target: y_binary = (y_true == c)
│  │
│  ├─ Fit: f_c = IsotonicRegression().fit(p_c, y_binary)
│  │
│  └─ Store calibrator f_c
│
└─ OUTPUT: {f_CN, f_MCI, f_AD}

TRANSFORM:
INPUT: p_test_uncal [m × 3]
│
├─ For each class c:
│  │
│  ├─ p_test_cal[:, c] = f_c.transform(p_test_uncal[:, c])
│  │
│  └─ Apply isotonic mapping
│
├─ Normalize: p_test_cal /= sum(p_test_cal, axis=1)
│
└─ OUTPUT: p_test_cal [m × 3]  (calibrated)
```

### 2. Cost-Sensitive Decision Rule

```
INPUT: p_calibrated [n × 3], Cost Matrix C [3 × 3]

Cost Matrix C:
         Pred: CN  MCI   AD
True CN   [ 0.0  0.3  0.9 ]
True MCI  [ 0.5  0.0  0.7 ]
True AD   [ 1.0  0.8  0.0 ]
│
├─ For each sample i:
│  │
│  ├─ For each possible prediction j ∈ {CN, MCI, AD}:
│  │  │
│  │  ├─ Compute expected cost:
│  │  │   Cost(j|x_i) = Σ_k C[k,j] * p_i[k]
│  │  │
│  │  │   Example: Cost(predict CN | x_i)
│  │  │   = C[CN,CN]*p[CN] + C[MCI,CN]*p[MCI] + C[AD,CN]*p[AD]
│  │  │   = 0.0*p[CN] + 0.5*p[MCI] + 1.0*p[AD]
│  │  │
│  │  └─ Store Cost(j|x_i)
│  │
│  ├─ Select: ŷ_i = argmin_j Cost(j|x_i)
│  │
│  └─ Store prediction
│
└─ OUTPUT: ŷ [n]  (cost-optimized predictions)
```

### 3. Expected Calibration Error (ECE)

```
INPUT: y_true [n], p_pred [n × 3]
│
├─ Extract confidence: conf = max(p_pred, axis=1)
├─ Extract predictions: ŷ = argmax(p_pred, axis=1)
├─ Compute accuracy: acc = (ŷ == y_true)
│
├─ Create bins: [0.0-0.1, 0.1-0.2, ..., 0.9-1.0]
│
├─ For each bin b:
│  │
│  ├─ Samples in bin: mask = (conf ∈ bin_b)
│  ├─ Bin accuracy: acc_b = mean(acc[mask])
│  ├─ Bin confidence: conf_b = mean(conf[mask])
│  ├─ Bin weight: w_b = count(mask) / n
│  │
│  └─ Bin error: w_b * |acc_b - conf_b|
│
├─ Sum all bin errors
│
└─ OUTPUT: ECE = Σ_b w_b * |acc_b - conf_b|
```

## 📏 Dimensioni dei Dati (Esempio Fold)

```
Dataset: ~500 samples total
│
├─ Training:     300 samples (60%)
│  ├─ Clinical features: [300 × ~50]
│  └─ Images: [300 × 1 × 128 × 128 × 50]
│
├─ Calibration:  100 samples (20%)
│  ├─ Used for: Isotonic Regression fitting
│  └─ Not used for: Model training
│
└─ Test:         100 samples (20%)
   ├─ Used for: Final evaluation
   └─ Predictions: [100 × 3] probabilities → [100] class labels
```

## ⚙️ Parametri Principali

```yaml
Model Training:
  epochs: 30
  batch_size: 16
  n_boosting_iterations: 8
  freeze_layers: 2

Calibration:
  method: Isotonic Regression
  n_classes: 3 (CN, MCI, AD)
  
Cost Matrix:
  AD→CN: 1.0  (most severe)
  CN→AD: 0.9  (very severe)
  AD→MCI: 0.8
  MCI→AD: 0.7
  MCI→CN: 0.5
  CN→MCI: 0.3  (least severe)

Evaluation:
  ece_bins: 10
  metrics: [Accuracy, Precision, Recall, F1, ECE, Cost]
```

## 🎯 Obiettivi Target

```
✅ Calibrazione (Isotonic Regression):
   ECE:  < 0.10  (ideale: 0.06-0.08)
   MCE:  < 0.15
   Improvement: > 30% rispetto a uncalibrated

✅ Cost Reduction:
   Mean Cost: < 0.35 per sample
   Reduction: > 15% rispetto a standard argmax
   
✅ Performance Maintainance:
   Accuracy drop: < 3%
   F1-Score: maintained or improved
   Sensitivity AD: improved (less missed AD cases)
```

## 🔀 Confronto: Standard vs Cost-Sensitive

```
┌─────────────────────────────────────────────────────────────┐
│           Standard (Argmax)    vs    Cost-Sensitive         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Decision Rule:                                             │
│    ŷ = argmax p_i            |    ŷ = argmin Σ C(i,j)·p_i  │
│                                                             │
│  Optimizes:                                                 │
│    Maximum probability       |    Minimum expected cost    │
│                                                             │
│  Behavior:                                                  │
│    Treats all errors equal   |    Prioritizes severe errors│
│                                                             │
│  Example (p = [0.4, 0.35, 0.25]):                          │
│                                                             │
│    Predicted: CN             |    Predicted: MCI           │
│    (highest prob)            |    (lowest cost)            │
│                                                             │
│  Cost Impact:                                               │
│    Higher avg cost           |    Lower avg cost ✓         │
│    More severe errors        |    Fewer severe errors ✓    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Output Files Schema

```
results/calibration_experiments/default_TIMESTAMP/
│
├── fold_results/
│   ├── fold_0/
│   │   ├── reliability_uncalibrated.png
│   │   ├── reliability_calibrated.png
│   │   ├── cost_matrix.png
│   │   ├── confusion_matrices_comparison.png
│   │   ├── fold_0_summary.json
│   │   └── fold_0_per_class_metrics.csv
│   │
│   ├── fold_1/
│   ├── fold_2/
│   ├── fold_3/
│   └── fold_4/
│
├── visualizations/
│   ├── metrics_comparison.png          (Accuracy, F1, etc. per fold)
│   ├── cost_reduction.png              (Cost standard vs cost-sens)
│   └── calibration_improvement.png      (ECE before vs after)
│
├── aggregated_metrics.csv              (Mean ± Std across folds)
└── summary_report.txt                  (Human-readable report)
```

---

## 🚦 Quick Decision Tree: Quale Setup Usare?

```
START: Vuoi calibrare le probabilità?
│
├─ SÌ → Quale modello?
│  │
│  ├─ CNN individuale (ResNet)
│  │  └─> USA: Temperature Scaling
│  │
│  └─ Ensemble finale (IRBoostSH)
│     └─> USA: Isotonic Regression ✓✓✓
│
└─ Vuoi decisioni cost-sensitive?
   │
   ├─ SÌ → Hai probabilità calibrate?
   │  │
   │  ├─ SÌ → USA: Cost-Sensitive Decision ✓
   │  │
   │  └─ NO → PRIMA calibra, POI applica cost-sensitive
   │
   └─ NO → Usa decisione standard (argmax)
```

---

**Nota:** Questo diagramma mostra il flusso completo implementato nei file creati. Per codice eseguibile, vedi `test_calibration.py` e `run_calibration_experiments.py`.
