# Probability Calibration and Cost-Sensitive Decision Making

## Panoramica

Questo modulo implementa una pipeline completa per la **calibrazione delle probabilità** e il **decision-making cost-sensitive** nel contesto della predizione clinica di Alzheimer (CN/MCI/AD).

### Motivazione Clinica

Nel dominio clinico, non tutti gli errori di classificazione hanno lo stesso impatto:
- **AD→CN** (demenza non diagnosticata): Errore gravissimo - paziente non riceve cure necessarie
- **CN→AD** (falso positivo demenza): Errore molto grave - stress psicologico, trattamenti inappropriati  
- **AD→MCI**: Sottostima della gravità, ritardo nel trattamento appropriato
- **MCI→AD**: Sovrastima, ma comunque nell'area di monitoraggio
- **CN↔MCI**: Errori meno gravi ma comunque clinicamente rilevanti

## Componenti Principali

### 1. Temperature Scaling (TS)

**Scopo**: Calibrare le probabilità di output delle reti neurali.

**Metodo**: Apprende un singolo parametro scalare $T$ (temperatura) che riscala i logits prima della softmax:

$$p_{\text{calibrated}} = \text{softmax}(\text{logits} / T)$$

**Quando usare**:
- Per calibrare output di CNN individuali (ResNet per immagini)
- Quando le probabilità della rete sono overconfident

**Esempio**:
```python
from src.predict.calibration import TemperatureScaling

ts = TemperatureScaling()
ts.fit(logits_calib, y_calib)
probs_calibrated = ts.transform(logits_test)
```

### 2. Isotonic Regression (IR)

**Scopo**: Calibrare le probabilità dell'insieme finale del modello multimodale.

**Metodo**: Apprende una trasformazione monotonica non-parametrica per ogni classe che mappa le probabilità predette alle frequenze osservate.

**Quando usare**:
- Per calibrare l'output finale dell'ensemble multimodale
- Più flessibile del Temperature Scaling
- **Raccomandato come approccio principale per il nostro caso**

**Esempio**:
```python
from src.predict.calibration import IsotonicRegressionCalibrator

ir = IsotonicRegressionCalibrator()
ir.fit(probs_calib, y_calib)
probs_calibrated = ir.transform(probs_test)
```

### 3. Cost-Sensitive Bayesian Decision Rule

**Scopo**: Selezionare la classe che minimizza il costo clinico atteso invece di massimizzare la probabilità.

**Metodo**: Data la matrice di costo $C(i,j)$ e le probabilità calibrate $p(y=i|x)$, seleziona:

$$\hat{y}(x) = \arg\min_j \sum_i C(i,j) \cdot p(y=i|x)$$

**Matrice di Costo Clinica** (CN/MCI/AD):

|           | Pred: CN | Pred: MCI | Pred: AD |
|-----------|----------|-----------|----------|
| True: CN  | 0.0      | 0.3       | **0.9**  |
| True: MCI | 0.5      | 0.0       | 0.7      |
| True: AD  | **1.0**  | 0.8       | 0.0      |

**Esempio**:
```python
from src.predict.calibration import CostSensitiveDecision

cost_decision = CostSensitiveDecision()
y_pred = cost_decision.predict(probs_calibrated)

# Con informazioni sui costi
y_pred, expected_costs = cost_decision.predict_with_costs(probs_calibrated)

# Valutare la riduzione di costo
cost_eval = cost_decision.evaluate_cost(y_true, y_pred)
```

## Workflow Completo

### Setup dello Split dei Dati

Il sistema riutilizza lo split già implementato per conformal prediction:

```python
from src.data.processing_data import processing_features_cv_with_calibration

folds = processing_features_cv_with_calibration(use_augmented=False)
# Ogni fold contiene:
# - X_train (60%), y_train
# - X_calib (20%), y_calib  <- usato per calibrazione
# - X_test (20%), y_test
```

### Pipeline di Calibrazione

```python
from src.predict.calibration import calibrate_multimodal_model

results = calibrate_multimodal_model(
    model=trained_model,           # Modello IRBoostSH già addestrato
    X_train=X_train, y_train=y_train,
    X_calib=X_calib, y_calib=y_calib,  # Set di calibrazione (20%)
    X_test=X_test, y_test=y_test,
    apply_temperature_scaling=False,   # Opzionale per CNN individuali
    apply_isotonic_regression=True,    # Raccomandato per ensemble
    use_cost_sensitive=True,           # Decision rule Bayesiana
    output_dir='results/calibration'
)

# Risultati
probs_calibrated = results['probs_calibrated']
y_pred_standard = results['predictions_standard']        # argmax
y_pred_cost_sens = results['predictions_cost_sensitive'] # costo minimo
calib_metrics = results['calibration_metrics']
cost_evaluation = results['cost_evaluation_cost_sensitive']
```

## Metriche di Valutazione

### Calibration Metrics

1. **Expected Calibration Error (ECE)**
   - Misura la differenza tra confidenza e accuratezza
   - Formula: $\text{ECE} = \sum_b \frac{n_b}{n} |acc_b - conf_b|$
   - **Lower is better** (0 = perfettamente calibrato)

2. **Maximum Calibration Error (MCE)**
   - Massima deviazione tra confidenza e accuratezza
   - Più sensibile a outlier rispetto a ECE

3. **Brier Score**
   - Mean squared error tra probabilità predette e one-hot encoding
   - Misura sia calibrazione che sharpness

4. **Negative Log-Likelihood (NLL)**
   - Standard loss probabilistica

### Cost Metrics

1. **Mean Cost per Sample**
   - Costo clinico medio per paziente
   - **Metrica principale per decision rule**

2. **Per-class Cost**
   - Costo medio per ciascuna classe vera
   - Identifica dove il modello è più costoso

3. **Cost Reduction**
   - Percentuale di riduzione rispetto a decisione standard (argmax)

## Visualizzazioni Generate

1. **Reliability Diagram** (before/after calibration)
   - Mostra calibrazione su diversi livelli di confidenza
   - Linea diagonale = calibrazione perfetta

2. **Cost Matrix Heatmap**
   - Visualizza la matrice di costo clinica

3. **Confusion Matrices** (standard vs cost-sensitive)
   - Confronta errori tra le due strategie di decisione

## Script di Test

### Test su Singolo Fold

```bash
python test_calibration.py --fold 0 --output-dir results/calibration_fold0
```

### Parametri disponibili

- `--fold N`: Fold da testare (default: 0)
- `--no-augmented`: Disabilita data augmentation
- `--output-dir PATH`: Directory per salvare risultati

### Output Generati

```
results/calibration_fold0/
├── reliability_uncalibrated.png       # Calibrazione prima di IR
├── reliability_calibrated.png         # Calibrazione dopo IR
├── cost_matrix.png                    # Matrice di costo clinica
├── confusion_matrices_comparison.png   # Standard vs cost-sensitive
├── fold_0_detailed_results.csv        # Predizioni e probabilità
└── fold_0_calibration_metrics.csv     # Metriche ECE, MCE, Brier, NLL
```

## Analisi di Sensibilità

Il sistema include analisi di sensibilità per valutare la stabilità delle decisioni rispetto a perturbazioni della matrice di costo:

```python
from src.predict.calibration import CostSensitiveDecision

cost_decision = CostSensitiveDecision()
sensitivity = cost_decision.sensitivity_analysis(
    probs_calibrated,
    perturbation_range=np.linspace(0.8, 1.2, 5)
)

# Mostra stabilità delle predizioni sotto perturbazioni ±20%
for factor, stability in zip(sensitivity['perturbation_factors'], 
                             sensitivity['prediction_stability']):
    print(f"Factor {factor:.2f}x: {stability*100:.1f}% unchanged")
```

## Integrazione con Esperimenti Esistenti

### Modificare run_experiments.py

```python
# Aggiungere calibrazione agli esperimenti esistenti
from src.predict.calibration import calibrate_multimodal_model

# Dopo training
results = calibrate_multimodal_model(
    model=model,
    X_train=X_train, y_train=y_train,
    X_calib=X_calib, y_calib=y_calib,
    X_test=X_test, y_test=y_test,
    apply_isotonic_regression=True,
    use_cost_sensitive=True,
    output_dir=f'results/experiment_suite/fold_{fold_idx}/calibration'
)

# Salvare metriche aggiuntive
experiment_results['calibration_ece'] = results['calibration_metrics']['After'][0]
experiment_results['mean_cost_standard'] = results['cost_evaluation_standard']['mean_cost_per_sample']
experiment_results['mean_cost_optimized'] = results['cost_evaluation_cost_sensitive']['mean_cost_per_sample']
```

## Riferimenti Teorici

1. **Temperature Scaling**:
   - Guo et al. "On Calibration of Modern Neural Networks", ICML 2017

2. **Isotonic Regression**:
   - Zadrozny & Elkan. "Transforming Classifier Scores into Accurate Multiclass Probability Estimates", KDD 2002

3. **Cost-Sensitive Learning**:
   - Elkan. "The Foundations of Cost-Sensitive Learning", IJCAI 2001

4. **Expected Calibration Error**:
   - Naeini et al. "Obtaining Well Calibrated Probabilities Using Bayesian Binning", AAAI 2015

## FAQ

**Q: Quale metodo di calibrazione usare?**  
A: Per il modello ensemble multimodale, **Isotonic Regression** è raccomandato. Temperature Scaling è più adatto per singole reti neurali.

**Q: Quanto grande deve essere il calibration set?**  
A: Usiamo 20% dei dati (dopo split train/test). È un buon compromesso tra training e calibrazione.

**Q: La matrice di costo è fissa?**  
A: Puoi modificarla in base a evidenze cliniche o preferenze. Il sistema include sensitivity analysis per valutare l'impatto.

**Q: Il decision rule cost-sensitive peggiora l'accuracy?**  
A: Possibilmente sì, ma riduce il **costo clinico totale**, che è più importante in contesti medici.

**Q: Posso usare questa pipeline con MC Dropout?**  
A: Sì! Le probabilità da MC Dropout possono essere calibrate con IR prima di applicare la decision rule cost-sensitive.

## Esempio Completo

```python
# 1. Load data
from src.data.processing_data import processing_features_cv_with_calibration
folds = processing_features_cv_with_calibration(use_augmented=False)
fold = folds[0]

# 2. Train model
from src.train.boosting import training_function
params = {"id": 1, "epochs": 30, "n_iteration": 8}
model = training_function(fold['X_train'], fold['y_train'], 0, params)

# 3. Calibrate and apply cost-sensitive decision
from src.predict.calibration import calibrate_multimodal_model
results = calibrate_multimodal_model(
    model=model,
    X_train=fold['X_train'], y_train=fold['y_train'],
    X_calib=fold['X_calib'], y_calib=fold['y_calib'],
    X_test=fold['X_test'], y_test=fold['y_test'],
    apply_isotonic_regression=True,
    use_cost_sensitive=True,
    output_dir='results/my_calibration_test'
)

# 4. Analyze results
print("ECE improvement:", 
      results['calibration_metrics'].loc[0, 'Improvement %'], "%")
print("Cost reduction:", 
      results['cost_evaluation_standard']['mean_cost_per_sample'] - 
      results['cost_evaluation_cost_sensitive']['mean_cost_per_sample'])
```

## Prossimi Passi

1. ✅ Implementare Temperature Scaling
2. ✅ Implementare Isotonic Regression
3. ✅ Implementare matrice di costo e decision rule
4. ✅ Creare metriche di valutazione
5. 🔲 Integrare con run_experiment_suite.py
6. 🔲 Analizzare risultati su tutti i fold
7. 🔲 Ottimizzare matrice di costo basandosi su risultati
8. 🔲 Documentare risultati finali
