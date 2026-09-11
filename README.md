# Northstar Model Validation

**Independent validation framework for a machine-learning credit risk model.**

Northstar is a portfolio project demonstrating how an independent Model Risk / Model Validation function can assess an ML-based credit risk model beyond a single predictive-performance number.

The current release validates a transparent **Logistic Regression default-probability model** built on the public Home Credit Default Risk application dataset.

> **Portfolio / educational project.** This repository is not a production credit decision system, regulatory model approval, or substitute for a bank's formal model governance process.

## Validation scope

Version 1.0 focuses on five validation areas:

| Validation area | V1.0 scope | Assessment |
|---|---|---|
| Data & prediction integrity | Target, IDs, coverage and consistency checks | **PASS** |
| Discrimination | ROC-AUC, PR-AUC, Gini, KS, risk ordering, thresholds | **ACCEPTABLE** |
| Calibration | Aggregate calibration, Brier score, decile analysis | **ACCEPTABLE** |
| Stability & robustness | Bootstrap and repeated subsample analysis | **PASS** |
| Segment performance | Performance across predefined population segments | **SCREENING SIGNAL** |

**Overall validation conclusion: ACCEPTABLE WITH MONITORING.**

This conclusion is explicitly limited to the validation scope above.

## Key validation results

Validation sample: **61,503 applications**, including **4,965 observed adverse outcomes**.

- ROC-AUC: **0.7483**
- PR-AUC: **0.2279**
- Gini: **0.4966**
- KS: **0.3715**
- Observed bad rate: **8.0728%**
- Mean predicted probability: **8.0425%**
- Calibration-in-the-large: **-0.0003**
- Brier score: **0.0685**
- Risk-band monotonicity: **9/9 adjacent bands**
- Highest-risk decile captures **32.49%** of observed defaults
- Lowest-risk decile default rate: **1.43%**
- Highest-risk decile default rate: **26.22%**
- Risk separation: **18.33x**

Bootstrap confidence intervals and repeated 90% subsample analysis indicate stable aggregate discrimination and calibration metrics.

Segment analysis identified **9 segments with degraded discrimination** under the project's predefined screening rule. These results are **screening signals, not statistical proof of model weakness**.

## Model

The model is intentionally simple and transparent:

- Logistic Regression
- median imputation for numeric variables
- standardization of numeric variables
- most-frequent imputation for categorical variables
- one-hot encoding of categorical variables
- fixed random state: 42
- validation split: stratified 80/20

The stored model package contains:

```text
model/model_package/
├── model.joblib
├── preprocessing.joblib
├── model_metadata.yaml
└── feature_dictionary.yaml
```

## Repository structure

```text
northstar-model-validation/
├── README.md
├── pyproject.toml
├── configs/
│   └── validation.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── model/
│   ├── train.py
│   ├── predict.py
│   ├── target_specification.yaml
│   └── model_package/
├── validation/
│   ├── data_quality.py
│   ├── reproducibility.py
│   ├── performance.py
│   ├── findings.py
│   ├── runner.py
│   └── legacy/
├── reports/
│   ├── Independent_Validation_Report_V1.0.md
│   ├── baseline/
│   ├── discrimination/
│   ├── calibration/
│   └── segments/
├── tests/
└── notebooks/
```

The `legacy/` directory contains the exploratory validation scripts used to establish and verify the V1.0 results. They are retained for provenance but are not part of the public framework API.

## Reproduction

Install the project:

```bash
python -m pip install -e ".[dev]"
```

Make sure the Home Credit application data is present under `data/raw/`.

Train the model package:

```bash
python model/train.py
```

Generate predictions:

```bash
python model/predict.py
```

Run the framework checks:

```bash
python -m validation.runner
```

Run automated tests:

```bash
pytest
```

## Data

The project uses the public **Home Credit Default Risk** dataset.

The repository documentation treats the dataset's `TARGET` as an observed adverse/default outcome. It is **not interpreted as a regulatory 12-month Probability of Default**, because the public source documentation does not provide sufficient information to establish that interpretation.

Raw data is intentionally excluded from Git version control.

See `data/README.md` for data provenance and handling notes.

## Limitations

V1.0 does not provide:

- temporal / out-of-time validation;
- production drift monitoring or PSI;
- formal statistical significance testing for the segment screening results;
- fairness assessment;
- challenger-model comparison;
- business/economic validation;
- production implementation validation;
- regulatory model approval.

These are explicit scope limitations, not claims that the model has passed those areas.

## Validation philosophy

Northstar separates:

1. **model performance** from
2. **model validation**, and
3. **model risk conclusions**.

A good aggregate metric does not by itself establish model validity.

The framework therefore considers discrimination, calibration, robustness, segmentation, reproducibility and documented limitations together.

## License

MIT
