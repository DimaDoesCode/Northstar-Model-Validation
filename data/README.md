# Data — Northstar Model Validation Framework

## 1. Dataset

The V0.1 implementation uses the publicly available:

**Home Credit Default Risk**

Source:

https://www.kaggle.com/competitions/home-credit-default-risk/data

The dataset was originally published by Home Credit as part of the
Home Credit Default Risk Kaggle competition.

The dataset contains information related to loan applications and
historical credit behaviour.

---

## 2. Purpose

The dataset is used as a synthetic/public proxy for a retail credit
risk modelling environment.

The purpose of this project is NOT to reproduce the original Kaggle
competition.

Instead, the data is used to demonstrate an independent model
validation process for a machine-learning-based credit risk model.

The validation framework is intended to assess:

- data quality;
- model reproducibility;
- conceptual soundness;
- predictive performance;
- calibration;
- stability;
- robustness;
- model limitations;
- validation findings.

---

## 3. Available Tables

The original dataset contains the following major tables:

### application_train.csv

The main training table.

Each row represents one loan application.

This table contains the target variable `TARGET`.

### application_test.csv

The corresponding application table without the target variable.

It is not used as an independent validation dataset in V0.1.

### bureau.csv

Historical credit information obtained from other financial
institutions for clients represented in the main application data.

### bureau_balance.csv

Monthly balance information for historical bureau credits.

### previous_application.csv

Previous Home Credit applications associated with clients
represented in the main application data.

### installments_payments.csv

Historical installment payment information.

### POS_CASH_balance.csv

Monthly snapshots of previous POS and cash loans.

### credit_card_balance.csv

Historical credit card balance information.

### HomeCredit_columns_description.csv

Description of variables contained in the dataset.

---

## 4. V0.1 Data Scope

V0.1 intentionally uses only:

    application_train.csv

The historical tables are excluded from the initial model.

This is a deliberate simplification.

The objective of V0.1 is to validate the architecture of the
independent validation framework before introducing complex
many-to-one feature engineering.

Historical tables may be incorporated in later versions.

---

## 5. Unit of Observation

The primary unit of observation is a loan application.

The application identifier is:

    SK_ID_CURR

One row corresponds to one application in `application_train.csv`.

---

## 6. Target Variable

The target variable is:

    TARGET

The original dataset defines `TARGET` as the observed default outcome
associated with the application.

For this project:

    TARGET = 1
        observed default / adverse outcome

    TARGET = 0
        no observed default / adverse outcome

Important:

The project does NOT interpret `TARGET` as a regulatory 12-month
Probability of Default.

The exact observation horizon and target construction used by the
original data provider are not sufficiently specified for that
interpretation.

Therefore the model is referred to as a:

    Default Probability Model

rather than:

    12-month PD Model

---

## 7. Data Leakage Considerations

Only information that would reasonably be available at or before the
loan application decision should be considered eligible for model
development.

Any variable representing information observed after the decision
date, directly or indirectly, must be treated as a potential source
of target leakage.

The validation framework will explicitly assess this issue.

---

## 8. Data Quality Considerations

The validation framework will assess at least:

- missing values;
- duplicated observations;
- invalid values;
- unexpected ranges;
- inconsistent data types;
- categorical value consistency;
- extreme observations;
- suspicious feature-target relationships.

Data quality tests are considered part of independent validation,
not merely preprocessing.

---

## 9. Dataset Limitations

This dataset is a public competition dataset and should not be
considered equivalent to a production banking dataset.

Known limitations include:

- incomplete documentation of the full data-generating process;
- competition-oriented dataset construction;
- limited information about production-time data availability;
- limited information about the exact target observation horizon;
- potential differences between competition data and real banking
  environments.

These limitations are part of the validation context.

---

## 10. Data Licensing

The dataset is subject to the terms and rules of the original
Home Credit Default Risk competition.

Users of this repository are responsible for complying with the
applicable dataset terms.

The repository does not redistribute the original dataset.

---

## 11. Local Directory Structure

The dataset should be placed locally as:

    data/
        raw/
            application_train.csv
            application_test.csv
            HomeCredit_columns_description.csv

Historical tables may be added later:

    data/
        raw/
            bureau.csv
            bureau_balance.csv
            previous_application.csv
            installments_payments.csv
            POS_CASH_balance.csv
            credit_card_balance.csv

Raw data must not be committed to Git.

---

## 12. Data Versioning

The project should record:

- dataset source;
- source URL;
- download date;
- file name;
- file size;
- SHA-256 checksum;
- number of rows;
- number of columns.

Example:

    dataset:
      name: Home Credit Default Risk
      source: Kaggle
      downloaded_at: YYYY-MM-DD

This information is intended to support reproducibility of the
validation process.