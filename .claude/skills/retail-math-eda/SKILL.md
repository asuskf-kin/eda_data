---
name: retail-math-eda
description: Conduct exploratory data analysis (EDA) on retail datasets combining rigorous mathematical and statistical methods with core retail business metrics. Use when analyzing retail sales, transaction logs, basket data, SKU performance, customer purchasing behavior, price elasticity, inventory turnover, or promotional effectiveness.
---

# Retail Math EDA

A comprehensive workflow for conducting exploratory data analysis on retail datasets, pairing
mathematical and statistical rigor with actionable retail business insights.

## When to Use

- Analyzing retail transactional data (point of sale, e-commerce, order lines).
- Evaluating customer purchase patterns, RFM distributions, and lifetime value dynamics.
- Conducting market basket analysis, product affinity, and cross-sell evaluation.
- Assessing pricing elasticity, promotional lift, and margin cannibalization.
- Investigating inventory dynamics, stockout impact, and demand variability (ABC-XYZ).

## Before Starting

Establish three things and write them down; every later step depends on them.

1. **Grain.** One row is what? Transaction x line item x SKU is the usual retail grain.
   Verify the candidate key is unique before trusting any aggregate.
2. **Observation cutoff `T_split`.** Fix the date at which customer-level features stop.
   Everything after it is outcome, never feature. See `references/gotchas.md`.
3. **Monetary convention.** Gross vs. net of returns, tax-in vs. tax-out, currency and FX date.
   State it once; a mixed convention invalidates every elasticity and margin number downstream.

The helper module `scripts/retail_eda.py` (Polars + NumPy + SciPy, no pandas required) implements
the computational parts of steps 1-4. Import it or copy the functions you need:

```python
import sys; sys.path.insert(0, "<skill_dir>/scripts")
import retail_eda as rx
```

Run `python scripts/retail_eda.py --self-test` to confirm the environment before touching real data.

## Workflow

### 1. Data Integrity and Missingness Diagnostics

- Inspect data structure, grain, and primary keys (e.g., Transaction ID + Line Item ID + SKU).
- Classify missing data mechanisms: MCAR, MAR, or MNAR. In retail, missing discount codes often
  mean zero discount, while missing customer IDs typically indicate guest checkouts. Encode that
  domain meaning explicitly instead of imputing.
- Detect structural anomalies: negative quantities (returns or adjustments), negative prices,
  duplicate transactions, and non-retail administrative records (internal testing accounts).
- Quantify transaction density across stores, channels, and time to identify system downtime or
  data collection outages. A store with zero rows for a week is almost never zero demand.

`rx.integridad(df, ...)` returns the null/negative/duplicate census; `rx.densidad_temporal(...)`
flags collection gaps.

### 2. Univariate Distribution and Heavy-Tail Analysis

- Calculate parametric and non-parametric summaries: mean, standard deviation, median, IQR,
  median absolute deviation (MAD), skewness, and excess kurtosis.
- Retail sales and customer spend exhibit heavy right tails (Pareto / power-law behavior). Test
  distributional fit with Kolmogorov-Smirnov and Anderson-Darling against log-normal, Pareto and
  Weibull. Estimate the tail index with the Hill estimator over the top-k order statistics.
- Separate zero-inflated distributions: analyze conversion (zero vs. non-zero spend) separately
  from the conditional spend distribution given purchase.
- Apply `log1p` or Box-Cox transformations to stabilize variance for linear downstream tasks,
  verifying the optimal lambda via maximum likelihood.

`rx.univariado(serie)` returns moments, robust stats, zero share, Hill alpha and fit tests.
Formulas and interpretation thresholds: `references/statistical-methods.md`.

### 3. Bivariate and Multivariate Dependency Modeling

- Assess correlation structures across monetary metrics, units and basket size with both linear
  and rank-based estimators:
  - Pearson for linear associations.
  - Spearman rank correlation and Kendall's tau for monotonic non-linear relationships.
  - Mutual Information (Kraskov-Stoegbauer-Grassberger estimator) for non-linear, non-monotonic
    dependence; fall back to a binned MI estimate when scikit-learn is unavailable.
- Evaluate significance with p-values adjusted for multiple testing via Benjamini-Hochberg FDR.
  With 150 columns you are running >11k pairwise tests; unadjusted p-values are meaningless there.
- Detect multivariate outliers using Mahalanobis distance (accounting for covariance) or Isolation
  Forests, distinguishing genuine VIP wholesale/B2B buyers from data entry errors. Confirm the
  distinction against a business attribute (customer type, tax ID) before dropping anything.
- Run PCA on standardized category sales vectors to identify dominant cross-category buying
  patterns, reading explained-variance ratios and the scree plot.

`rx.dependencias(df, cols)` returns the correlation panel with BH-adjusted q-values;
`rx.mahalanobis(df, cols)` returns distances and the chi-square cutoff.

### 4. Retail Business Dimensions and Commercial Unit Economics

#### A. Basket and Assortment Dynamics (Market Basket Analysis)

- Build item co-occurrence matrices from the transaction-item bipartite graph.
- Compute association rule metrics:
  - Support: `P(A and B)`
  - Confidence: `P(B|A) = Support(A and B) / Support(A)`
  - Lift: `P(A and B) / (P(A) * P(B))`. Focus on `Lift > 1.2` with significance via Fisher's exact test.
  - Leverage and conviction to measure deviation from independence.
- Identify anchor products (high support, low lift) versus affinity drivers (moderate support,
  high lift). They lead to opposite merchandising actions.

`rx.market_basket(df, txn_col, sku_col, top_n=...)`.

#### B. Customer Cohort and RFM Dynamics

- Compute Recency (days since last purchase), Frequency (distinct purchase days/orders) and
  Monetary value (total spend), all strictly at or before `T_split`.
- Check the probabilistic modeling foundations (Pareto/NBD or BG/NBD):
  - Transaction frequency against Poisson assumptions (dispersion test).
  - Monetary value conditional on frequency against Gamma-Gamma assumptions (independence of
    frequency and average order value).
- Perform cohort survival analysis with Kaplan-Meier curves to assess retention and locate churn
  inflection points across monthly/quarterly cohorts.

`rx.rfm(df, cliente, fecha, monto, t_split)`.

#### C. Price Elasticity and Promotional Effectiveness

- Model own-price elasticity of demand: `e = (%dQ) / (%dP)`.
- Fit log-log regressions: `ln(Q_it) = a + b*ln(P_it) + g*Controls + e`, where `b` is the elasticity.
- Verify homoscedasticity with Breusch-Pagan or White; apply Huber-White (HC1) robust standard
  errors when heteroscedasticity is present.
- Decompose sales into baseline versus promotional uplift: baseline is expected sales at regular
  price; incremental uplift is observed minus baseline.
- Identify cannibalization: negative cross-price elasticity on substitute SKUs during promotions.

`rx.elasticidad(df, precio, cantidad, controles=...)` returns beta, HC1 standard errors and the
Breusch-Pagan statistic. Read `references/gotchas.md` on endogeneity before reporting a number.

#### D. Assortment and Inventory ABC-XYZ Analysis

- **ABC (revenue contribution):** Pareto curve over SKUs sorted by cumulative revenue.
  A = top 80%, B = next 15%, C = remaining 5%.
- **XYZ (demand predictability):** coefficient of variation `CV = sigma / mu` of per-period demand.
  - X: `CV < 0.5` — steady, predictable.
  - Y: `0.5 <= CV <= 1.0` — variable but seasonal.
  - Z: `CV > 1.0` — sporadic, lumpy.
- Combine into the 9-box matrix (AX, AY, AZ, BX, BY, BZ, CX, CY, CZ) to guide stocking, promotion
  and markdown strategy.

`rx.abc_xyz(df, sku, ingreso, periodo)`.

### 5. Temporal Stochastic Decomposition and Calendar Effects

- Decompose aggregate sales into trend, seasonal and residual components with STL (Seasonal and
  Trend decomposition using Loess).
- Test stationarity on the differenced series with ADF and KPSS. The two tests answer opposite
  null hypotheses; report both and act only when they agree.
- Model calendar covariates: day-of-week seasonality (weekend vs. weekday), holiday lead-lag
  windows, and pay-cycle effects (bimonthly/monthly wages, payday spikes).
- Analyze stockout contamination: distinguish true zero demand from unrecorded demand caused by
  zero on-shelf inventory. Censored zeros bias every elasticity and forecast downward.

STL/ADF/KPSS require `statsmodels`; see `references/statistical-methods.md` for the calls and for
the censoring correction.

### 6. Executive Synthesis and Strategic Bridge

- Translate findings into executive metrics: GMROI, Average Order Value (AOV), Units Per
  Transaction (UPT) and Sell-Through Rate. Definitions: `references/retail-metrics.md`.
- Bridge statistical evidence to operational action:
  - Assortment rationalization (pruning CZ SKUs with negative contribution margin).
  - Cross-merchandising layout changes based on verified association rules.
  - Dynamic markdown timing derived from decay parameters and elasticity estimates.

Every recommendation should carry the statistic behind it, its uncertainty, and the one assumption
that would overturn it if wrong.

## Gotchas

- **Treating returns as standard negative sales.** Never compute variance or log transforms on raw
  negative values. Split returns into a distinct refund sub-analysis.
- **Endogeneity in price elasticity.** Price changes correlate with seasonal peaks and clearance.
  A positive own-price elasticity estimate is a specification failure, not a finding.
- **Data leakage in customer aggregates.** Always enforce the `T_split` cutoff when building RFM.
- **Aggregating without weighting.** Store-level averages cannot be averaged across stores. Weight
  by volume or footfall or you will produce Simpson's Paradox.
- **Confusing correlation with association.** High co-occurrence between two bestsellers reflects
  high marginal probabilities, not affinity. Always read Lift alongside Support.

Full worked explanations of each: `references/gotchas.md`.

## Reference Files

- `references/statistical-methods.md` — formulas, estimators, test procedures, thresholds.
- `references/retail-metrics.md` — commercial metric definitions and unit economics.
- `references/gotchas.md` — the five traps, why they happen, and how to detect them.
- `scripts/retail_eda.py` — Polars/NumPy/SciPy implementation of the computational steps.
