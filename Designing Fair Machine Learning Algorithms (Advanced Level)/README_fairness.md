# Fairness Evaluation and Mitigation in Machine Learning

This repository demonstrates how to measure and mitigate unfairness in binary classification models using the [Fairlearn](https://fairlearn.org/) library. The focus is on assessing disparities across sensitive groups (e.g., `SEX`) and applying mitigation strategies such as **postprocessing with `ThresholdOptimizer`** to improve fairness.

## 📌 Key Features

- Evaluate model fairness using:
  - False Positive Rate (FPR)
  - False Negative Rate (FNR)
  - Balanced Accuracy
- Compute **standard errors** for fairness metrics
- Visualize group-level disparities with **error bars**
- Apply **postprocessing mitigation** using `ThresholdOptimizer`
- Support for `equalized_odds` constraint to balance FPR and FNR across groups

---

## 📁 Project Structure

```
.
├── compute_error_metric.py   # Custom error computation for FPR, FNR, balanced accuracy
├── fairness_metrics.py       # Fairness metric dictionary used in MetricFrame
├── visualize_metrics.py      # Error bar plot function for group metrics
├── mitigate_postprocess.py   # ThresholdOptimizer setup and predictions
├── model_training.py         # Classifier training and test predictions
├── main_analysis.ipynb       # Jupyter notebook orchestrating full workflow
└── README.md                 # This file
```

---

## 📊 Fairness Metrics & Evaluation

Fairness is evaluated using the `MetricFrame` object from Fairlearn:

```python
metricframe_unmitigated = MetricFrame(
    metrics=fairness_metrics,
    y_true=y_test,
    y_pred=Y_pred,
    sensitive_features=A_test,
)
```

### Reported Metrics:
- `balanced_accuracy`
- `false_positive_rate`
- `false_negative_rate`

Standard errors are computed for uncertainty quantification.

### Visualization:
Group-level metrics with confidence intervals:

```python
plot_group_metrics_with_error_bars(
    metricframe_unmitigated, 
    "false_positive_rate", 
    "false_positive_error"
)
```

---

## 🛠️ Fairness Mitigation - Postprocessing

We apply **postprocessing** using Fairlearn's `ThresholdOptimizer`:

```python
postprocess_est = ThresholdOptimizer(
    estimator=estimator,
    constraints="equalized_odds",
    objective="balanced_accuracy_score",
    prefit=True,
    predict_method="predict_proba",
)
postprocess_est.fit(X=X_train, y=y_train, sensitive_features=A_train)
postprocess_pred = postprocess_est.predict(X_test, sensitive_features=A_test)
```

This adjusts group-specific thresholds to optimize performance while reducing disparities.

---

## 🧪 Assumptions

- **FPR and FNR have equal cost** across groups.
- In production scenarios, you may apply **weighting schemes** to reflect different societal or financial consequences.

---

## 📦 Requirements

- Python 3.7+
- scikit-learn
- fairlearn
- matplotlib
- numpy
- pandas

Install dependencies:

```bash
pip install fairlearn scikit-learn matplotlib numpy pandas
```

---

## 👩🏾‍🔬 Authors & Acknowledgements

This work builds on the fairness principles outlined in the Fairlearn documentation and academic literature such as [Hardt et al., 2016](https://arxiv.org/abs/1610.02413).