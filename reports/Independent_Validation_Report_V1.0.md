# NORTHSTAR MODEL VALIDATION
## Independent Validation Report

**Model:** Logistic Regression Credit Risk Model  
**Validation Version:** 1.0  
**Validation Sample:** 61,503 observations  
**Target:** `TARGET` — credit default indicator  
**Validation Scope:** Baseline Performance, Discrimination, Calibration, Stability & Robustness, Segment Performance

---

# 1. Executive Summary

This report presents an independent validation of a baseline Logistic Regression credit risk model developed using the Home Credit dataset.

The objective of the validation was to assess the model from several independent perspectives rather than relying on a single predictive performance metric. The validation framework covered:

1. Baseline model performance and data integrity;
2. Discriminatory power;
3. Probability calibration;
4. Statistical and sample-composition stability;
5. Performance across predefined population segments.

The validation was performed on a hold-out validation sample containing **61,503 observations**, including **4,965 defaults**, corresponding to an observed default rate of **8.07%**.

Overall, the model demonstrates **moderate discriminatory power, generally good calibration, and stable aggregate performance**. The risk ranking is strongly monotonic across risk deciles, and the model separates the lowest- and highest-risk deciles by a factor of approximately **18.3x** in observed default rate.

The principal model-risk observation concerns **segment-level discrimination**. Performance degradation was identified in nine of twenty predefined segments, concentrated primarily in the `EXT_SOURCE_2` and `EXT_SOURCE_3` variables. However, no segment met the configured threshold for severe degradation, and the segment analysis was explicitly designed as a descriptive screening procedure rather than a statistical significance test.

Accordingly, the validation does **not identify evidence of a material model failure within the tested scope**. The segment findings should nevertheless be treated as an area for further investigation and monitoring.

### Overall validation conclusion

**The model is considered acceptable within the scope of this validation, subject to the limitations and monitoring recommendations described in this report.**

The conclusion should not be interpreted as evidence that the model is production-ready for a regulated credit decisioning environment. Important areas such as temporal stability, population drift, implementation controls, fairness assessment, economic/business validation, and independent challenger modelling were outside the scope of this V1.0 validation.

---

# 2. Validation Objective

The purpose of independent validation is to determine whether the model performs as expected and whether there are identifiable sources of model risk that could materially affect its intended use.

The validation therefore considers several distinct questions:

- Does the model produce valid and complete predictions?
- Does the model discriminate between default and non-default observations?
- Are predicted probabilities reasonably aligned with observed default frequencies?
- Are aggregate performance measures stable under resampling and changes in sample composition?
- Does model performance deteriorate materially in identifiable population segments?

The validation intentionally avoids treating any single metric as sufficient evidence of model quality.

---

# 3. Model and Dataset Overview

## 3.1 Dataset

The source dataset contains:

- **307,511 observations**
- **122 columns**
- target variable: `TARGET`

The full population contains:

- `TARGET = 0`: **282,686 observations (91.93%)**
- `TARGET = 1`: **24,825 observations (8.07%)**

The identifier `SK_ID_CURR` was excluded from model features.

After identifier removal, the modelling dataset contained **120 features**:

- **104 numerical features**
- **16 categorical features**

## 3.2 Validation Sample

The validation sample contains:

- **61,503 observations**
- **56,538 non-defaults**
- **4,965 defaults**
- default rate: **8.0728%**

The observed target prevalence is consistent with the full dataset and training sample.

Prediction coverage was **100%**, with:

- zero missing predictions;
- unique prediction identifiers;
- target values restricted to `{0, 1}`;
- predicted probabilities in the range **0.000000–0.756372**.

---

# 4. Validation Methodology

The validation framework consists of five complementary analytical blocks.

| Validation area | Primary objective |
|---|---|
| Baseline Model | Establish model and data integrity and baseline performance |
| Discrimination | Assess ranking ability and separation of defaults/non-defaults |
| Calibration | Assess correspondence between predicted PD and observed default rates |
| Stability & Robustness | Assess sensitivity to resampling and sample composition |
| Segment Performance | Screen for localized performance degradation |

The tests are complementary. A model can, for example, demonstrate strong discrimination while simultaneously exhibiting poor calibration. Similarly, good aggregate performance does not guarantee consistent performance across population segments.

---

# 5. Baseline Model Validation

## 5.1 Baseline Performance

A Logistic Regression model was trained using a preprocessing pipeline containing the 120 non-ID features.

Validation performance was:

| Metric | Result |
|---|---:|
| ROC-AUC | **0.7483** |
| PR-AUC | **0.2279** |
| Gini | **0.4966** |
| KS | **0.3715** |

These results establish a moderate level of discriminatory performance.

The model's coefficients were also inspected as a basic interpretability and implementation sanity check. The largest absolute coefficients were associated with variables including `AMT_GOODS_PRICE`, `AMT_CREDIT`, `ORGANIZATION_TYPE`, `DAYS_EMPLOYED`, `EXT_SOURCE_2`, and `EXT_SOURCE_3`.

Coefficient inspection is not considered a substitute for formal feature-level validation; it is used here as a supporting diagnostic.

## 5.2 Threshold Observation

At a technical probability threshold of 0.50, the resulting confusion matrix was:

| | Predicted non-default | Predicted default |
|---|---:|---:|
| Actual non-default | 56,493 | 45 |
| Actual default | 4,905 | 60 |

This threshold produces a very small flagged population.

The threshold result is therefore **not interpreted as a recommended business decision threshold**. Business threshold selection requires economic loss functions, approval constraints, portfolio objectives, and policy considerations that are outside this validation scope.

---

# 6. Discrimination Validation

## 6.1 Overall Discrimination

The model achieved:

| Metric | Estimate | 95% CI |
|---|---:|---:|
| ROC-AUC | **0.748310** | **0.741786–0.755654** |
| PR-AUC | **0.227914** | **0.218486–0.237690** |
| Gini | **0.496620** | — |
| KS | **0.371547** | **0.359000–0.386093** |

Confidence intervals were estimated using **1,000 stratified bootstrap resamples**.

The relatively narrow confidence intervals indicate that the estimated aggregate discrimination metrics are statistically well-defined for this validation sample.

The PR-AUC is approximately **2.82 times the portfolio default prevalence**, providing additional evidence that the model has meaningful ranking ability in the context of the imbalanced target.

## 6.2 Risk-Band Analysis

The validation sample was divided into ten approximately equal-sized risk bands.

Observed default rates increased monotonically from:

- **1.4307%** in the lowest-risk decile;
- to **26.2234%** in the highest-risk decile.

All nine adjacent risk-band comparisons were non-decreasing:

**9/9 monotonic pairs = 100%.**

The ratio between the highest- and lowest-risk observed default rates is approximately:

**18.33x.**

The highest-risk decile contains **32.49% of all observed defaults** while representing approximately 10% of the validation population.

This provides strong evidence that the model produces a meaningful risk ordering.

## 6.3 Segment Discrimination

The discrimination analysis also evaluated twelve predefined categorical segments.

Segment ROC-AUC ranged from:

- minimum: **0.707567**
- mean: **0.739527**
- maximum: **0.749871**

The weakest segment was:

`NAME_INCOME_TYPE = Pensioner`

with ROC-AUC **0.707567**, representing an absolute difference of **0.040743** relative to the overall AUC.

The segment results do not by themselves establish a model weakness because segment size, prevalence, and statistical uncertainty must be considered.

---

# 7. Calibration Validation

## 7.1 Overall Calibration

The validation sample had:

- observed default rate: **8.0728%**
- mean predicted probability: **8.0425%**

The resulting calibration-in-the-large was:

**−0.000303**

This indicates very little aggregate bias between the mean predicted probability and the observed portfolio default rate.

Additional metrics were:

| Metric | Result |
|---|---:|
| Calibration-in-the-large | **−0.000303** |
| Mean absolute calibration error | **0.002292** |
| Maximum absolute calibration error | **0.007844** |
| Weighted calibration error | **0.000303** |
| Brier score | **0.068518** |
| Log loss | **0.249197** |

Overall, these results indicate close agreement between aggregate predicted and observed default probabilities.

## 7.2 Calibration by Risk Decile

Nine of ten risk deciles had observed default rates whose 95% confidence intervals contained the corresponding mean predicted probability.

Therefore:

- **9/10 deciles: inside 95% CI**
- **1/10 deciles: outside 95% CI**

The only deviation occurred in risk decile 8.

For this decile:

- mean predicted probability: **10.2562%**
- observed default rate: **11.0407%**
- calibration error: **+0.7844 percentage points**

This represents a localized calibration deviation rather than broad portfolio-level miscalibration.

The confidence intervals are used as a diagnostic tool and are **not interpreted as a formal hypothesis test of calibration**.

---

# 8. Stability and Robustness Validation

Two complementary approaches were used to assess stability:

1. bootstrap resampling;
2. repeated 90% subsampling.

## 8.1 Bootstrap Stability

The bootstrap analysis used:

**1,000 iterations**, all of which were valid.

Results:

| Metric | Mean | Std | 95% CI |
|---|---:|---:|---:|
| ROC-AUC | **0.748347** | 0.003575 | **0.741549–0.755271** |
| PR-AUC | **0.227980** | 0.005462 | **0.217399–0.238845** |
| Brier | **0.068495** | 0.000834 | **0.066945–0.070190** |

The bootstrap distributions remain tightly concentrated around the baseline estimates.

## 8.2 Subsample Stability

Repeated 90% subsamples were evaluated using **100 valid iterations**.

Results:

| Metric | Mean | Std | Maximum deviation |
|---|---:|---:|---:|
| ROC-AUC | **0.748387** | 0.001103 | **0.002970** |
| PR-AUC | **0.228141** | 0.001812 | **0.004356** |
| Brier | **0.068509** | 0.000286 | **0.000796** |

The relatively small deviations from baseline indicate low sensitivity of aggregate model performance to moderate changes in sample composition.

## 8.3 Stability Conclusion

Within the tested scope, the model demonstrates **stable aggregate discrimination and calibration-related metrics**.

No evidence of material instability was observed under bootstrap resampling or 90% subsampling.

The analysis does not establish temporal stability or resistance to population drift.

---

# 9. Segment Performance Validation

## 9.1 Methodology

Twenty predefined segments were constructed using quartiles of five continuous model-related variables:

- `AMT_INCOME_TOTAL`
- `AMT_CREDIT`
- `DAYS_BIRTH`
- `EXT_SOURCE_2`
- `EXT_SOURCE_3`

All segments satisfied the minimum data requirements:

- segment N ≥ 1,000;
- default count ≥ 30.

Therefore, no segment was classified as insufficiently populated.

## 9.2 Findings

The analysis identified:

- **11 segments — OK**
- **9 segments — DEGRADED**
- **0 segments — SEVERE DEGRADATION**
- **0 segments — INSUFFICIENT DATA**

The degradation was strongly concentrated in two variables.

| Feature | Segments | Degraded |
|---|---:|---:|
| `AMT_INCOME_TOTAL` | 4 | **0** |
| `AMT_CREDIT` | 4 | **0** |
| `DAYS_BIRTH` | 4 | **1** |
| `EXT_SOURCE_2` | 4 | **4** |
| `EXT_SOURCE_3` | 4 | **4** |

The weakest segment was:

`EXT_SOURCE_2 = (0.664, 0.855]`

with:

- N = **15,343**
- bad rate = **3.7737%**
- ROC-AUC = **0.698553**
- AUC difference versus overall = **−0.049757**

The most degraded `EXT_SOURCE_3` segment had:

- N = **12,329**
- bad rate = **7.2674%**
- ROC-AUC = **0.703341**
- AUC difference = **−0.044969**

No segment crossed the configured severe-degradation threshold of **−0.05**.

## 9.3 Interpretation

The segment analysis should be interpreted as a **screening mechanism**.

The configured classification rules were:

- AUC difference > −0.02 → OK
- AUC difference ≤ −0.02 → DEGRADED
- AUC difference ≤ −0.05 → SEVERE DEGRADATION

The analysis itself does not test statistical significance.

Consequently, the nine degraded segments should not be interpreted as nine confirmed model-risk issues.

Instead, they identify a concentrated area for further investigation, particularly around the relationship between model discrimination and the `EXT_SOURCE_2` / `EXT_SOURCE_3` feature ranges.

---

# 10. Consolidated Validation Findings

The validation produced the following principal findings.

### Finding 1 — Adequate aggregate discrimination

The model achieves ROC-AUC **0.7483** and KS **0.3715**, with relatively narrow bootstrap confidence intervals.

**Assessment: Acceptable within scope.**

### Finding 2 — Strong risk ordering

Observed default rates increase monotonically across all ten risk bands.

The highest-risk decile has an observed default rate of **26.22%**, compared with **1.43%** in the lowest-risk decile.

**Assessment: Positive validation evidence.**

### Finding 3 — Generally good calibration

The difference between observed portfolio default rate and mean predicted probability is approximately **0.03 percentage points**.

Nine of ten risk deciles fall within the corresponding 95% confidence intervals.

**Assessment: Generally acceptable, with one localized deviation.**

### Finding 4 — Stable aggregate performance

Bootstrap and 90% subsampling analyses show limited variation in ROC-AUC, PR-AUC and Brier score.

**Assessment: No material aggregate instability identified.**

### Finding 5 — Segment degradation signal

Nine of twenty continuous-variable segments were classified as degraded, with all four `EXT_SOURCE_2` segments and all four `EXT_SOURCE_3` segments showing degradation.

However:

- no segment showed severe degradation;
- all segments had sufficient observations and defaults;
- the analysis was descriptive;
- statistical significance was not established.

**Assessment: Screening signal requiring monitoring/further investigation, not confirmed model failure.**

---

# 11. Model Risk Assessment

Based on the evidence generated by the V1.0 validation framework, the model does not demonstrate a material weakness in its aggregate predictive performance.

The principal model-risk consideration is **localized discrimination degradation across segments defined by `EXT_SOURCE_2` and `EXT_SOURCE_3`**.

This finding is potentially relevant because these variables appear to identify regions of the population where the model's ability to rank risk is weaker.

However, the available evidence is insufficient to determine whether this represents:

- statistical sampling variation;
- genuine heterogeneous model performance;
- feature interaction effects;
- differences in population composition;
- or an economically material weakness.

The current validation therefore classifies this as a **model-risk screening signal rather than a confirmed model-risk finding**.

---

# 12. Recommendations

## Recommendation 1 — Monitor segment performance

Future validation cycles should monitor discrimination in the segments associated with `EXT_SOURCE_2` and `EXT_SOURCE_3`.

Particular attention should be given to whether the observed degradation persists across independent samples.

## Recommendation 2 — Investigate the identified segments if materiality warrants

If the model is intended for material credit decisioning, the degraded segments should be subject to additional statistical and business/materiality analysis before relying on the result as evidence of model weakness.

## Recommendation 3 — Revalidate calibration periodically

The overall calibration result is strong, but the localized deviation in risk decile 8 should be monitored in subsequent validation cycles.

## Recommendation 4 — Establish temporal monitoring

The current validation does not test temporal stability.

For a production credit-risk model, subsequent validation should include out-of-time performance and population stability analysis.

---

# 13. Validation Limitations

The conclusions in this report are subject to the following limitations.

### 13.1 Single validation sample

The principal analysis uses one hold-out validation sample.

### 13.2 No temporal validation

The current framework does not establish performance stability over time or under changing economic conditions.

### 13.3 No population drift analysis

PSI, characteristic drift, and other population stability measures were not included in V1.0.

### 13.4 No formal statistical significance testing of segment degradation

Segment performance classifications are descriptive and screening-oriented.

### 13.5 No fairness assessment

The current framework does not constitute a formal fairness, bias, or protected-class assessment.

### 13.6 No challenger model

The validation does not compare the model against an independently developed challenger or alternative modelling methodology.

### 13.7 No business/economic validation

Threshold optimisation, expected loss, cost-sensitive decisions, profitability, approval strategy, and portfolio economics are outside the current scope.

### 13.8 No production implementation validation

The current analysis validates the modelling outputs available in the validation environment. It does not constitute a full production implementation, deployment, or code-change control review.

---

# 14. Overall Validation Conclusion

The Northstar baseline Logistic Regression credit risk model was subjected to five complementary validation analyses covering baseline integrity, discriminatory power, calibration, aggregate stability, and segment performance.

The evidence indicates that:

- the model produces complete and valid predictions for the validation sample;
- aggregate discriminatory power is moderate and statistically stable;
- risk ordering across deciles is strongly monotonic;
- predicted probabilities are generally well calibrated;
- aggregate performance is robust to bootstrap resampling and moderate changes in sample composition;
- localized segment-level discrimination degradation exists, particularly in segments defined by `EXT_SOURCE_2` and `EXT_SOURCE_3`.

No severe segment degradation was identified, and the observed segment findings do not establish statistical significance or material model risk within the current scope.

### Final validation assessment

**ACCEPTABLE WITH MONITORING**

The model is considered acceptable within the scope of the V1.0 validation framework.

The principal condition is continued monitoring and, where appropriate, further investigation of the identified segment-level discrimination degradation.

This conclusion is limited to the validation scope described in this report and should not be interpreted as a full regulatory, production-readiness, or enterprise Model Risk Management approval.

---

# 15. Validation Matrix

| Validation dimension | Result | Assessment |
|---|---|---|
| Data / prediction integrity | 100% prediction coverage; no missing predictions | **PASS** |
| Baseline discrimination | ROC-AUC 0.7483; KS 0.3715 | **ACCEPTABLE** |
| Risk ordering | 9/9 monotonic risk-band transitions | **PASS** |
| Discrimination statistical stability | Narrow 95% bootstrap CIs | **PASS** |
| Calibration | MAE 0.0023; 9/10 deciles within CI | **ACCEPTABLE** |
| Calibration exception | Decile 8 outside 95% CI | **OBSERVATION** |
| Aggregate robustness | Small bootstrap/subsample deviations | **PASS** |
| Segment performance | 9/20 degraded; 0 severe | **SCREENING SIGNAL** |
| Overall validation | No material weakness identified within scope | **ACCEPTABLE WITH MONITORING** |

---

# 16. Appendix — Key Validation Metrics

### Dataset

- Total observations: **307,511**
- Features after ID removal: **120**
- Validation observations: **61,503**
- Validation defaults: **4,965**
- Validation default rate: **8.0728%**

### Discrimination

- ROC-AUC: **0.748310**
- PR-AUC: **0.227914**
- Gini: **0.496620**
- KS: **0.371547**
- ROC-AUC 95% CI: **0.741786–0.755654**
- PR-AUC 95% CI: **0.218486–0.237690**
- KS 95% CI: **0.359000–0.386093**

### Risk bands

- Number of bands: **10**
- Monotonic transitions: **9/9**
- Lowest-risk default rate: **1.4307%**
- Highest-risk default rate: **26.2234%**
- Risk separation ratio: **18.33x**
- Top-decile default capture: **32.49%**

### Calibration

- Mean predicted probability: **8.0425%**
- Calibration-in-the-large: **−0.000303**
- Brier score: **0.068518**
- Log loss: **0.249197**
- Mean absolute calibration error: **0.002292**
- Maximum absolute calibration error: **0.007844**
- Deciles within 95% CI: **9/10**

### Robustness

- Bootstrap iterations: **1,000**
- Valid bootstrap iterations: **1,000**
- Subsample fraction: **90%**
- Valid subsample iterations: **100**
- Maximum ROC-AUC subsample deviation: **0.002970**
- Maximum PR-AUC subsample deviation: **0.004356**
- Maximum Brier subsample deviation: **0.000796**

### Segment performance

- Total analysed segments: **20**
- OK: **11**
- Degraded: **9**
- Severe degradation: **0**
- Insufficient data: **0**
- Weakest AUC: **0.698553**
- Weakest feature: **EXT_SOURCE_2**
- No severe degradation threshold breached

---

**End of Report — Northstar Model Validation V1.0**