# Conformal Prediction per Uncertainty Quantification

## 🎯 Metodo Implementato: Inductive Conformal Prediction

### Referenze Scientifiche
- **Olsson et al. (2022)**: "Estimating diagnostic uncertainty in AI-assisted cancer diagnosis" [Nature Medicine]
- **Sarica et al. (2024)**: Brain age estimation with conformal prediction
- **Vovk et al. (2005)**: "Algorithmic Learning in a Random World" (libro fondamentale su CP)

### Vantaggi vs Tree Subsampling
| Aspetto | Tree Subsampling | Conformal Prediction |
|---------|------------------|---------------------|
| **Garanzie matematiche** | ❌ Nessuna | ✅ Coverage garantita (1-α) |
| **Distribution-free** | ❌ Dipende da RF | ✅ Funziona con QUALSIASI modello |
| **Correlazione con errori** | ⚠️ Empirica | ✅ GARANTITA matematicamente |
| **Applicabilità clinica** | ❌ Solo RF | ✅ Tutti i modelli (RF, CNN, etc.) |
| **Interpretabilità** | ⚠️ "Variazione tra alberi" | ✅ "Set di diagnosi plausibili" |
| **Retraining richiesto** | ❌ No | ❌ No (solo calibration set) |

## 📊 Come Funziona

### Step 1: Training (INVARIATO)
```python
# Train model normalmente su 60% del dataset
model.fit(X_train, y_train)
```

### Step 2: Calibration (NUOVO - 20% del training originale)
```python
# Su calibration set, calcola non-conformity scores
for sample in calibration_set:
    score = 1 - P(true_class)  # Quanto il modello era "sbagliato"
    
# Threshold = quantile(scores, 1-α)
# α = 0.1 → 90% coverage guarantee
```

### Step 3: Prediction (TEST)
```python
# Per ogni test sample:
prediction_set = {classes where P(class) ≥ 1 - threshold}

# Uncertainty = (|prediction_set| - 1) / (n_classes - 1)
# Range: [0, 1]
#   0.0 = singleton set {AD} → completamente certo
#   0.5 = doublet set {AD, MCI} → incerto tra 2 diagnosi
#   1.0 = full set {CN, MCI, AD} → completamente incerto
```

## 💡 Output Clinico

### Esempio 1: Alta Certezza
```
Paziente #123
  Prediction set: {AD}
  Uncertainty: 0.00
  → Diagnosi certa: Alzheimer's Disease
```

### Esempio 2: Incertezza Moderata
```
Paziente #456
  Prediction set: {MCI, AD}
  Uncertainty: 0.50
  → Incerto tra MCI e AD → richiede follow-up
```

### Esempio 3: Alta Incertezza
```
Paziente #789
  Prediction set: {CN, MCI, AD}
  Uncertainty: 1.00
  → Completamente incerto → necessari test aggiuntivi
```

## 🔬 Requisiti per Implementazione Completa

### 1. Modificare `processing_data.py`
Serve aggiungere split calibration:
```python
def processing_features_cv(use_augmented=False):
    # Attualmente: 80% train, 20% test
    # NUOVO: 60% train, 20% calibration, 20% test
    
    for fold in folds:
        X_train, y_train = ...
        X_calib, y_calib = ...  # NUOVO: 25% del train originale
        X_test, y_test = ...
```

### 2. Modificare `evaluate_predict.py`
Aggiungere parametro `use_conformal`:
```python
def evaluate(model, X_train, X_calib, y_calib, X_test, y_test, 
             use_mcdo=False, use_tta=False, use_conformal=False):
    
    if use_conformal:
        conf_results = model.predict_with_conformal(
            X_test, X_calib, y_calib, alpha=0.1
        )
        
        # Metriche:
        # - conformal_uncertainty: [0, 1]
        # - prediction_sets: lista di set per ogni paziente
        # - coverage: % pazienti con true_label in prediction_set
```

### 3. Aggiungere a `config.yaml`
```yaml
exp12:
  id: 12
  train: false
  reuse_from: exp1
  params:
    use_conformal: true
    conformal_alpha: 0.1  # 90% coverage
  description: 'Conformal Prediction for guaranteed UQ'
```

## 📈 Metriche da Tracciare

### 1. Coverage (CRITICO - deve essere ≥ 1-α)
```python
coverage = true_labels in prediction_sets
# Target: ≥ 90% per α=0.1
# Se < 90% → calibration set troppo piccolo o distributional shift
```

### 2. Conformal Uncertainty
```python
uncertainty = (|set| - 1) / (n_classes - 1)
# Distribuzione:
#   - 0.0-0.2: Prediction certa (singleton)
#   - 0.3-0.7: Incertezza moderata (doublet/triplet)
#   - 0.8-1.0: Alta incertezza (quasi tutti i classes)
```

### 3. Correlazione con Errori
```python
# Calcola Spearman ρ tra uncertainty e prediction errors
# Expected: ρ > 0.3 (forte correlazione)
# CP garantisce matematicamente questa correlazione
```

### 4. Set Size Distribution
```python
print(f"Singleton sets: {sum(|set| == 1) / n_samples:.1%}")
print(f"Doublet sets: {sum(|set| == 2) / n_samples:.1%}")
print(f"Full sets: {sum(|set| == n_classes) / n_samples:.1%}")

# Ideale:
# - 70-80% singleton (certo)
# - 15-25% doublet (incerto tra 2)
# - <5% full (completamente incerto)
```

## ⚙️ Parametri da Ottimizzare

### 1. Alpha (Miscoverage Rate)
```python
alpha = 0.1   # 90% coverage (default raccomandato)
alpha = 0.05  # 95% coverage (più conservativo, set più grandi)
alpha = 0.2   # 80% coverage (più aggressivo, set più piccoli)
```

### 2. Calibration Set Size
```python
# Minimo: 50-100 samples per classe
# Raccomandato: 20% del training set
# Più grande = coverage più stabile
```

## 🚀 Next Steps

1. **Modificare processing_data.py** per creare calibration split
2. **Aggiungere use_conformal a evaluate_predict.py**
3. **Creare exp12 con conformal prediction**
4. **Confrontare con MCDO e TTA**:
   - Epistemic (MCDO): model uncertainty
   - Aleatoric (TTA): data uncertainty
   - Conformal: **guaranteed** prediction uncertainty

## 📊 Confronto Completo dei Metodi

| Metodo | Tipo Incertezza | Output | Garanzie | Retraining |
|--------|----------------|--------|----------|------------|
| **MC Dropout** | Epistemic (model) | Score [0-1] | ❌ Nessuna | ❌ No |
| **TTA** | Aleatoric (data) | Score [0-1] | ❌ Nessuna | ❌ No |
| **Conformal** | Total (prediction) | Set + Score | ✅ Coverage (1-α) | ❌ No (solo calib) |

**Recommendation**: Usare **tutti e 3** per full uncertainty decomposition:
- Epistemic alto → model non ha visto abbastanza casi simili
- Aleatoric alto → dati clinici sono rumorosi/incerti
- Conformal alto → la predizione è intrinsecamente incerta (multiple diagnosi plausibili)

## ✅ Vantaggi Clinici di Conformal Prediction

1. **Risk Stratification**:
   - Singleton set → basso rischio diagnostico
   - Multiple classes → alto rischio → richiede conferma

2. **Interpretabilità**:
   - "Questo paziente ha probabilmente MCI o AD, non possiamo escludere nessuno dei due"
   - vs "uncertainty = 0.42" (meno interpretabile)

3. **Garanzie Matematiche**:
   - Coverage garantita → ≥90% pazienti hanno true_label nel set
   - Distribution-free → funziona sempre, indipendentemente dal modello

4. **FDA/Regulatory**:
   - Metodo statisticamente robusto con garanzie dimostrabili
   - Citato in letteratura medica (Olsson et al., Nature 2022)
