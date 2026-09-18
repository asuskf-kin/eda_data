# Statistical Methods Reference

Formulas, estimators and decision thresholds for the mathematical half of the workflow.
Everything here assumes a cleaned frame at a declared grain, with returns already separated.

---

## 1. Missingness mechanisms

| Mechanism | Definition | Retail example | Handling |
|---|---|---|---|
| MCAR | `P(missing) ⟂ everything` | Random POS packet loss | Complete-case analysis is unbiased but wasteful |
| MAR | `P(missing)` depends on observed data | Loyalty ID missing more often in express lanes | Model-based imputation conditioned on the observed driver |
| MNAR | `P(missing)` depends on the unobserved value itself | Cost field blank exactly on negative-margin SKUs | Cannot be imputed away; model the missingness indicator |

**Test for MCAR:** Little's MCAR test, or the cheap proxy — build a binary indicator `is_missing(col)`
and test its association with every other column (chi-square for categorical, Mann-Whitney for
numeric). Any strong association rejects MCAR.

**Retail-specific meanings that are NOT missing data:**
- Missing discount code → discount = 0.
- Missing customer ID → guest checkout, a real segment, not an unknown customer.
- Missing cost → usually a master-data gap, not a zero cost. Never let it become zero margin.

---

## 2. Univariate statistics

### Moments and robust analogues

```
mean      mu    = (1/n) Σ x_i
sd        sigma = sqrt( (1/(n-1)) Σ (x_i - mu)^2 )
CV              = sigma / mu                       (requires mu > 0)
median    m     = Q2
IQR             = Q3 - Q1
MAD             = median(|x_i - m|)
MAD_n           = 1.4826 * MAD                     (consistent with sigma under normality)
skewness  g1    = m3 / m2^(3/2)
excess kurt g2  = m4 / m2^2 - 3
```

Use `MAD_n` and IQR, not `sigma`, as the scale for any retail monetary variable. The variance of a
heavy-tailed sample is dominated by the few largest observations and is not stable across resamples.

### Heavy-tail diagnosis

Retail spend, basket value and SKU velocity are usually log-normal in the body and Pareto in the
tail. Establish which:

**Hill estimator** for the tail index over the top `k` order statistics `x_(1) ≥ ... ≥ x_(k+1)`:

```
alpha_hat(k) = k / Σ_{i=1..k} ( ln x_(i) - ln x_(k+1) )
```

Plot `alpha_hat(k)` against `k` (the Hill plot) and read the value off the plateau. Rules of thumb:

| alpha_hat | Meaning |
|---|---|
| `< 1` | Infinite mean. Sample means are meaningless; report medians and totals only |
| `1 – 2` | Finite mean, infinite variance. Never report a standard deviation or a t-test |
| `2 – 4` | Finite variance, heavy tail. CLT works but converges slowly; bootstrap CIs |
| `> 4` | Effectively light-tailed for practical purposes |

**Fit tests.** Kolmogorov-Smirnov is weak in the tail (its statistic is dominated by the centre);
Anderson-Darling weights the tails and is the right choice here.

```python
from scipy import stats
stats.kstest(x, "lognorm", args=stats.lognorm.fit(x, floc=0))
stats.anderson(np.log(x), dist="norm")      # AD for log-normality
stats.kstest(x, "pareto", args=stats.pareto.fit(x, floc=0))
stats.kstest(x, "weibull_min", args=stats.weibull_min.fit(x, floc=0))
```

With `n > 50_000` every goodness-of-fit test rejects. Subsample to ~5k for the p-value and use the
*statistic* (or the log-likelihood / AIC across candidate families) to rank the fits.

### Zero-inflation

Do not model a zero-inflated spend variable as one distribution. Split it:

```
P(spend > 0)                        → conversion / incidence model (Bernoulli)
f(spend | spend > 0)                → conditional severity model (log-normal or Gamma)
E[spend] = P(spend>0) * E[spend|>0]
```

For count data (units, visits) test the split formally: fit Poisson, negative binomial (NB2) and
zero-inflated Poisson, and compare by likelihood-ratio / Vuong. Overdispersion `Var > Mean` is
almost universal in retail counts, so Poisson is nearly always wrong.

### Variance-stabilizing transforms

- `log1p(x)` — safe with structural zeros, no parameter to estimate, interpretable as approximate
  percent change. The default.
- **Box-Cox** — requires `x > 0`; the MLE for lambda:

```python
from scipy import stats
xt, lam = stats.boxcox(x[x > 0])
```

`lam ≈ 0` → use the log. `lam ≈ 0.5` → square root. `lam ≈ 1` → no transform needed.
Report the lambda and its 95% CI (`stats.boxcox_normplot` / profile likelihood); if the CI contains
0, prefer the log for interpretability.

- **Yeo-Johnson** when the column legitimately contains zeros or negatives (e.g. net margin).

---

## 3. Dependency estimators

| Estimator | Captures | Robust to outliers | Retail use |
|---|---|---|---|
| Pearson `r` | Linear | No | Only after transformation |
| Spearman `rho` | Monotonic | Yes | Default for raw monetary columns |
| Kendall `tau-b` | Monotonic, ties-aware | Yes | Small samples, many ties (ratings, tiers) |
| Mutual information | Any dependence | Moderate | Non-monotonic (e.g. price vs. units at multiple price points) |
| Distance correlation | Any dependence, = 0 iff independent | Moderate | Confirmation when MI is ambiguous |

**KSG mutual information** (Kraskov-Stoegbauer-Grassberger), k-nearest-neighbour estimator:

```python
from sklearn.feature_selection import mutual_info_regression
mi = mutual_info_regression(X, y, n_neighbors=3, random_state=0)
```

MI is in nats and unbounded. Normalize for comparability:

```
MI_norm = MI / min( H(X), H(Y) )        in [0, 1]
```

Without scikit-learn, fall back to a binned plug-in estimate on quantile bins (10–20 bins), and
subtract the Miller-Madow bias correction `(B_x*B_y - B_x - B_y + 1) / (2n)`.

**Interpretation trap:** `MI > 0` with `|rho| ≈ 0` is the interesting case — a real non-monotonic
relationship. `MI > 0` with `|rho| ≈ 1` adds nothing over the correlation.

### Multiple testing — Benjamini-Hochberg

Sort the `m` p-values ascending. Find the largest `k` with `p_(k) ≤ (k/m) * q`. Reject all
hypotheses up to `k`.

```python
from scipy.stats import false_discovery_control
qvals = false_discovery_control(pvals, method="bh")
```

Use `q = 0.05` for confirmatory claims, `q = 0.10` for exploratory screening. Report the q-value,
not the raw p-value, in any table with more than ~20 rows.

### Mahalanobis distance

```
D2(x) = (x - mu)^T * S^-1 * (x - mu)
```

Under multivariate normality `D2 ~ chi2(p)`. Flag `D2 > chi2.ppf(0.975, p)`.

Two cautions:
1. The classical `mu` and `S` are themselves destroyed by the outliers you are hunting (masking).
   Use the Minimum Covariance Determinant estimator (`sklearn.covariance.MinCovDet`) for a robust
   version, or compute on log-transformed columns.
2. `S` is singular when columns are collinear (e.g. revenue, units and price all present). Drop the
   redundant column or use the pseudo-inverse.

**Isolation Forest** as the non-parametric alternative: no distributional assumption, handles mixed
scales, `contamination` set from the business expectation of B2B share, not tuned to fit.

### PCA on category vectors

Build the customer x category (or store x category) matrix of sales shares, standardize columns,
then:

```python
from sklearn.decomposition import PCA
p = PCA().fit(Z)
p.explained_variance_ratio_
```

- Standardize, or the largest category dominates PC1 mechanically.
- Use **shares within row** (row sums to 1) when you want buying *mix*; use absolute values when you
  want buying *size*. PC1 on absolute values is almost always just "total spend".
- Keep components by the scree elbow or the Kaiser criterion (eigenvalue > 1), not by a fixed 80%.
- Compositional data (shares) violates PCA assumptions; apply a centred log-ratio (CLR) transform
  first if the shares are the analysis object.

---

## 4. Association rules

For items `A` and `B` over `N` transactions, with counts from the 2x2 table:

|  | B | not B |
|---|---|---|
| **A** | `n11` | `n10` |
| **not A** | `n01` | `n00` |

```
Support(A∩B)   = n11 / N
Support(A)     = (n11 + n10) / N
Confidence     = Support(A∩B) / Support(A)
Lift           = Support(A∩B) / (Support(A) * Support(B))
Leverage       = Support(A∩B) - Support(A) * Support(B)
Conviction     = (1 - Support(B)) / (1 - Confidence)
```

- **Lift = 1** → independence. **Lift > 1.2** with significance → actionable affinity.
- **Leverage** is the absolute-scale complement of Lift: a pair can have Lift 4 on 11 baskets.
  Rank by leverage when you need volume impact, by lift when you need the strongest signal.
- **Conviction** → ∞ as confidence → 1; it measures how often the rule would be wrong if A and B
  were independent. Asymmetric, unlike lift.

**Significance:** Fisher's exact test on the 2x2 table (`scipy.stats.fisher_exact`), then BH across
all candidate pairs. With thousands of SKUs, restrict to the top-N by support before testing —
otherwise the multiple-testing correction annihilates everything real.

**Anchor vs. affinity driver:**

| | Low lift | High lift |
|---|---|---|
| **High support** | Anchor (bread, milk) — traffic driver, do not relocate | Category staple pair — protect adjacency |
| **Low support** | Noise | Affinity driver — cross-merchandising candidate |

---

## 5. RFM and probabilistic CLV foundations

```
Recency   R = (T_split - last_purchase_date).days
Frequency F = count of distinct purchase occasions in [T_start, T_split]
Monetary  M = total (or mean) spend in [T_start, T_split]
```

The buy-till-you-die models rest on assumptions you should check *before* fitting anything:

**Pareto/NBD and BG/NBD assume:**
1. While active, purchases follow a Poisson process with rate `lambda`.
   → Check: per-customer dispersion. `Var(F) / Mean(F)` across customers should exceed 1 (that is
   the Gamma heterogeneity), but *within* a customer the inter-purchase times should be roughly
   exponential. Test with a KS test on normalized inter-purchase gaps.
2. `lambda` is Gamma-distributed across customers.
   → Check: the frequency histogram should be negative-binomial shaped, not bimodal. Bimodality
   means you have two populations (retail + B2B) and must segment before fitting.
3. Dropout is independent of purchasing.
   → Check: correlation of `F` with `R` conditional on tenure. A strong negative relationship beyond
   what the model implies signals a lifecycle effect the model will not capture.

**Gamma-Gamma (monetary) additionally assumes** average order value is independent of frequency.
→ Check: Spearman correlation between `F` and `M/F`. `|rho| > 0.1` invalidates it; use a frequency-
conditional monetary model instead.

### Kaplan-Meier retention

```
S(t) = Π_{t_i ≤ t} ( 1 - d_i / n_i )
```

where `d_i` are churn events at `t_i` and `n_i` the at-risk set. Greenwood's formula gives the
variance. In retail, "churn" is not observed — define it explicitly (e.g. no purchase in
`2 * median inter-purchase interval`) and state the definition alongside every curve.

Right-censor every customer at `T_split`. Compare cohorts with the log-rank test; if curves cross,
the log-rank test is invalid — use restricted mean survival time instead.

---

## 6. Price elasticity

### Specification

```
ln(Q_it) = a + b * ln(P_it) + g' * X_it + u_i + v_t + e_it
```

`b` is the own-price elasticity directly (constant-elasticity form). `u_i` are SKU fixed effects,
`v_t` time fixed effects. Controls `X` should include at minimum: promotion flag, display/feature
flag, competitor price if available, and a seasonality term.

Expected sign: **negative**. A positive `b` means the price variation in your data is driven by
demand, not supply — see `gotchas.md`.

### Heteroscedasticity

**Breusch-Pagan:** regress `e_hat^2` on the regressors; `LM = n * R^2 ~ chi2(k)`.
**White:** same, with squares and cross-products added — catches misspecification too, at the cost
of many degrees of freedom.

If rejected (it usually is), keep OLS point estimates and switch to **Huber-White HC1**:

```
Var_HC1(b) = (n / (n-k)) * (X'X)^-1 * ( Σ e_i^2 x_i x_i' ) * (X'X)^-1
```

With panel data (SKU x week), cluster by SKU instead — serial correlation within a SKU is a bigger
problem than raw heteroscedasticity.

### Cross-price elasticity and cannibalization

```
e_AB = d ln(Q_A) / d ln(P_B)
```

`e_AB > 0` → substitutes (a promotion on B steals from A: cannibalization).
`e_AB < 0` → complements (a promotion on B lifts A: halo).

Estimate on the promotion windows only, with the non-promoted period as the baseline, and always
report the *net* category effect: category lift = own-SKU uplift − Σ cannibalized volume.

### Baseline and uplift decomposition

```
Baseline_t    = expected units at regular price, no promo    (fit on non-promo periods only)
Uplift_t      = Observed_t - Baseline_t
Lift %        = Uplift_t / Baseline_t
```

Fit the baseline with a model that has no access to the promotion flag, then predict *into* the
promotion window. Fitting on the full series lets the promotion contaminate the baseline and
understates the uplift.

---

## 7. ABC-XYZ classification

**ABC** — sort SKUs by revenue descending, take the cumulative share:

| Class | Cumulative revenue |
|---|---|
| A | 0 – 80% |
| B | 80 – 95% |
| C | 95 – 100% |

**XYZ** — per-period demand `CV = sigma / mu` over a consistent calendar (weekly is standard):

| Class | CV | Demand character |
|---|---|---|
| X | `< 0.5` | Steady, forecastable |
| Y | `0.5 – 1.0` | Variable, seasonal |
| Z | `> 1.0` | Sporadic, lumpy |

Two requirements that are routinely violated:
1. **Include zero-demand periods** in the CV. Dropping them makes every intermittent SKU look
   steady, which is exactly backwards.
2. **Use a fixed window** (e.g. the last 52 weeks) for every SKU. A SKU launched 6 weeks ago has no
   comparable CV; classify it as "new" and exclude it.

For genuinely intermittent demand, CV is a poor descriptor — use the Syntetos-Boylan classification
on average inter-demand interval (ADI) and squared CV instead:
`ADI > 1.32 and CV2 > 0.49` → lumpy; `ADI > 1.32 and CV2 ≤ 0.49` → intermittent.

**The 9-box actions:**

| | X (steady) | Y (seasonal) | Z (lumpy) |
|---|---|---|---|
| **A** | Automate replenishment, tight safety stock | Forecast with seasonality, pre-build | High service level, high safety stock, manual review |
| **B** | Periodic review | Seasonal buy | Make-to-order or drop-ship |
| **C** | Min-max, low attention | Consider delisting after season | Delist candidate — check margin contribution first |

---

## 8. Time series decomposition

### STL

```python
from statsmodels.tsa.seasonal import STL
res = STL(y, period=7, robust=True).fit()      # daily data, weekly seasonality
res.trend, res.seasonal, res.resid
```

- `robust=True` downweights holiday spikes so they do not distort the seasonal component.
- Retail usually needs **two** seasonalities (weekly + annual). STL handles one; use MSTL
  (`statsmodels.tsa.seasonal.MSTL`) with `periods=(7, 365)` for both.
- Decompose on `log(y)` when the seasonal amplitude grows with the level (multiplicative seasonality),
  which is the retail norm.

### Stationarity

| Test | Null hypothesis | Reject means |
|---|---|---|
| ADF | Unit root (non-stationary) | Stationary |
| KPSS | Stationary (around level or trend) | Non-stationary |

```python
from statsmodels.tsa.stattools import adfuller, kpss
adfuller(y, autolag="AIC")
kpss(y, regression="c", nlags="auto")
```

Read them jointly:

| ADF | KPSS | Conclusion |
|---|---|---|
| Reject | Fail to reject | Stationary. Proceed |
| Fail to reject | Reject | Unit root. Difference once, retest |
| Reject | Reject | Heteroscedastic or structural break. Detrend, do not difference |
| Fail to reject | Fail to reject | Not enough data / low power. Do not conclude |

### Calendar covariates

Encode explicitly rather than hoping a seasonal term absorbs them:

- Day of week (6 dummies) — payday and weekend effects.
- Payday proximity: days since the 15th and since month-end, or a triangular kernel around each.
- Holiday **windows**, not points: lead (-7..-1), day, lag (+1..+3). Pre-holiday pull-forward and
  post-holiday trough are separate effects with opposite signs.
- Moving holidays (Easter, Ramadan, Lunar New Year) need a date table, not a month dummy.
- Trading-day count per month when working monthly — a month with 5 Saturdays is not seasonal noise.

### Stockout censoring

An observed zero is either true zero demand or censored demand. If you have on-hand inventory:

```
censored_t = (units_sold_t == 0) and (on_hand_t <= 0)
```

Treat censored periods as missing (not zero) when computing CV, when fitting the baseline, and when
estimating elasticity. Without inventory data, a practical proxy: a zero flanked by non-zero demand
in a SKU whose ADI is otherwise short is suspicious; flag it rather than silently trusting it.
Ignoring censoring biases demand estimates downward and makes the SKU look like a delist candidate
when the real problem was availability.

---

## 9. Weighting and aggregation

Any metric that is a ratio must be aggregated as a **ratio of sums**, never a mean of ratios:

```
WRONG:  AOV_chain = mean(AOV_store)
RIGHT:  AOV_chain = sum(revenue) / sum(orders)
```

When a simple mean is genuinely wanted (e.g. "the typical store"), state the weight explicitly
(orders, footfall, or selling area) and report both the weighted and unweighted figure when they
disagree — the disagreement is itself the finding.
