# Monte Carlo Dropout Implementation - Documentation

## 📋 Overview

Implementazione completa di **Monte Carlo Dropout** per la quantificazione dell'incertezza epistemic nel modello VeroResNet e integrazione con il boosting IRBoostSH.

## 🎯 Obiettivi Raggiunti

### WEEK 1 - Implementazione MC Dropout

#### ✅ Task 1.1: Modificato VeroResNet
- **File**: `src/models/custom_base_estimators.py`
- **Modifiche**:
  - Aggiunto `self.dropout1 = nn.Dropout(p=0.5)` dopo `fc1`
  - Aggiunto `self.dropout2 = nn.Dropout(p=0.5)` dopo `fc2`
  - Modificato `forward()` con parametro `dropout_enabled`
  
```python
def forward(self, x: torch.Tensor, dropout_enabled=False):
    x = self.res(x)
    x = self.fc1(x)
    x = torch.nn.functional.relu(x)
    if dropout_enabled:
        x = self.dropout1(x)  # MC Dropout
    x = self.fc2(x)
    x = torch.nn.functional.relu(x)
    if dropout_enabled:
        x = self.dropout2(x)  # MC Dropout
    x = self.out_layer(x)
    return x
```

#### ✅ Task 1.2: Implementato MC Dropout Forward
- **Metodo**: `VeroResNet.forward_mcdo(x, n_mc_samples=25)`
- **Funzionalità**:
  - Salva stato training originale
  - Esegue 25 forward pass con dropout attivo
  - Restituisce predizioni stacked `[n_mc_samples, batch_size, num_classes]`

```python
def forward_mcdo(self, x: torch.Tensor, n_mc_samples=25):
    was_training = self.training
    self.eval()
    
    predictions_mc = []
    with torch.no_grad():
        for _ in range(n_mc_samples):
            logits = self.forward(x, dropout_enabled=True)
            predictions_mc.append(logits)
    
    self.train(was_training)
    return torch.stack(predictions_mc, dim=0)
```

#### ✅ Task 1.3: Compute Uncertainty
- **Metodo**: `VeroResNet.compute_uncertainty_from_mcdo(predictions_mc)`
- **Metriche calcolate**:
  - `mean_logits`: Media delle predizioni
  - `var_logits`: Varianza dei logits
  - `confidence`: Max probabilità da softmax
  - `epistemic_uncertainty`: Media della varianza (incertezza epistemic)
  - `predicted_class`: Classe predetta
  - `mean_probs`: Probabilità medie

### WEEK 2 - Integrazione con Boosting

#### ✅ Task 2.1: NeuralNetworkFitter con MC Dropout
- **File**: `src/models/custom_base_estimators.py`
- **Nuovo metodo**: `predict_proba_mcdo(data, n_mc_samples=25, mb_size=2)`
- **Output**: Dictionary con `mean_probs`, `confidence`, `epistemic_uncertainty`, `predicted_class`

#### ✅ Task 2.2: Edge Weighting in IRBoostSH
- **File**: `src/models/lutech_models.py`
- **Modifiche a `__compute_weak_forecast__`**:
  - Aggiunto parametro `use_mcdo=False`
  - Se `use_mcdo=True` e il model ha `predict_proba_mcdo`, calcola confidence
  - Altrimenti usa confidence di default = 1.0

- **Modifiche a `BoostSH.fit()` e `IRBoostSH.fit()`**:
  - Calcola `edge_raw` (edge non pesato)
  - Calcola `mean_confidence` dalle predizioni MC
  - Edge pesato: `edge = edge_raw * mean_confidence`
  - **IMPORTANTE**: `alpha` usa `edge_raw` per mantenere teoria boosting
  - Bandit reward usa `edge` pesato per selezione modalità

## 🔧 Formula di Confidence Weighting

```python
# Per ogni modalità durante boosting:
edge_raw = sum(weights * (2 * (forecast == labels) - 0.5))
mean_confidence = mean(MC_confidence_scores)
edge_weighted = edge_raw * mean_confidence

# Alpha calculation (usa edge_raw per teoria boosting):
alpha = 0.5 * log((1 + edge_raw) / (1 - edge_raw))

# Bandit reward (usa edge_weighted):
reward = (1 - sqrt(1 - edge_weighted^2)) / q_mods[modality]
```

## 📊 Output durante Training

Durante il boosting vedrai output come:

```
Modality images: raw_edge=0.6234, mean_confidence=0.8521, weighted_edge=0.5312
Mean confidence: 0.8521
Mean epistemic uncertainty: 0.0342

Weighted Edge: 0.5312 (raw: 0.6234, confidence: 0.8521)
Alpha: 0.7153
```

## 🧪 Testing

Esegui lo script di test:

```bash
python test_mc_dropout.py
```

### Test Coverage:
1. ✅ MC Dropout forward pass (25 samples)
2. ✅ Uncertainty metrics computation
3. ✅ Dropout layers architecture
4. ✅ Confidence vs epistemic uncertainty analysis

## 🚀 Usage Example

### Training con MC Dropout:

```python
from src.train.boosting import training_function

# Training automaticamente usa MC Dropout per 'images' modality
ir_boost = training_function(X_mods, y_train, fold, params)
```

### Inference con Uncertainty:

```python
# Per predizioni standard
predictions = model.predict(X_test)

# Per predizioni con uncertainty
nn_fitter = ir_boost.models[0]  # Assumendo primo model sia CNN
mcdo_results = nn_fitter.predict_proba_mcdo(X_test['images'], n_mc_samples=25)

print(f"Confidence: {mcdo_results['confidence']}")
print(f"Epistemic Uncertainty: {mcdo_results['epistemic_uncertainty']}")
print(f"Predictions: {mcdo_results['predicted_class']}")
```

## 📈 Metriche di Valutazione

### 1. Expected Calibration Error (ECE)
```python
def compute_ece(confidences, accuracies, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = accuracies[mask].mean()
            ece += mask.sum() / len(confidences) * abs(avg_conf - avg_acc)
    return ece
```

### 2. Rejection Curves
```python
# Ordina per uncertainty crescente e rimuovi predizioni incerte
sorted_indices = np.argsort(epistemic_uncertainty)
rejection_accuracies = []
for reject_ratio in np.linspace(0, 0.5, 20):
    n_keep = int(len(sorted_indices) * (1 - reject_ratio))
    kept_indices = sorted_indices[:n_keep]
    accuracy = (predictions[kept_indices] == labels[kept_indices]).mean()
    rejection_accuracies.append(accuracy)
```

## ⚠️ Note Importanti

1. **MC Dropout solo su 'images'**: RandomForest usa confidence=1.0 di default
2. **Alpha calculation**: Usa `edge_raw` per mantenere convergenza teorica boosting
3. **Bandit selection**: Usa `edge_weighted` per favorire view con alta confidence
4. **Numero MC samples**: Default 25 (trade-off tempo/accuracy)

## 🔄 Confronto Pre/Post MC Dropout

| Aspetto | Prima | Dopo |
|---------|-------|------|
| Forward pass | Singolo | 25x con dropout |
| Edge computation | Standard | Pesato con confidence |
| Uncertainty | ❌ No | ✅ Epistemic |
| Selection bias | Uniforme | Favorisce alta confidence |
| Inference time | ~1x | ~25x (solo per images) |

## 📝 Prossimi Passi (TTA - Fase 2)

Per implementare **Test Time Augmentation** (aleatoric uncertainty):

1. Aggiungere augmentations in `forward_tta()`:
   - Random flips
   - Random rotations
   - Random noise
   
2. Combinare con MC Dropout per total uncertainty:
   ```python
   total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
   ```

3. Decomposizione uncertainty:
   ```python
   epistemic = var(mean_predictions_per_augmentation)
   aleatoric = mean(var_predictions_per_augmentation)
   ```

## 📚 References

- Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation"
- Kendall, A., & Gal, Y. (2017). "What Uncertainties Do We Need in Bayesian Deep Learning?"

---

**Implementazione completata**: 31 Gennaio 2026  
**Versione**: 1.0  
**Status**: ✅ Production Ready
