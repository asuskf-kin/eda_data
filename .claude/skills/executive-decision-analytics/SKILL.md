---
name: executive-decision-analytics
description: Perform rigorous technical, financial, and statistical analyses to support executive decision-making and briefs. Evaluates metric tree root-cause decomposition (Price-Volume-Mix, bridge models), counterfactual causal impact, Monte Carlo scenario simulation, Expected Monetary Value (EMV) decision matrices, and parameter sensitivity (tornado analysis) to feed directly into executive briefs and strategic decision gates.
---

# Executive Decision Analytics

A technical analytical engine that provides the empirical evidence, stochastic modeling and
variance decomposition that justify an executive brief and the decision it asks for.

## When to Use

- Quantifying the exact financial exposure or commercial upside behind an executive hook.
- Root Cause Analysis on anomalous metric shifts — PVM decomposition, KPI driver tree traversal.
- Evaluating competing strategic alternatives under uncertainty via Monte Carlo and decision trees (EMV).
- Parameter sensitivity analysis (tornado charts, elasticity bounds) to identify the operational levers.
- Validating causal attribution (Difference-in-Differences, synthetic controls) against a correlational spike.
- Preparing verified inputs, scenario trade-offs and risk bounds for the Decision Matrix in
  `executive-brief-hook`.

## Before Starting

Establish two things and write them down; every number downstream inherits them.

1. **The comparison basis.** Actual vs. budget, vs. prior period, or vs. a control group. A variance
   is meaningless until the counterfactual is named.
2. **The monetary convention.** Gross vs. net, margin vs. revenue, currency and FX date, and whether
   figures are run-rate or period totals. State it once, at the top.

If the underlying data cannot support a monetary figure, say so and anchor the hook on a verified
operational count instead. A fabricated currency figure in a decision memo is worse than an honest
unit count — see **Gotchas**.

## Analytical Workflow

### 1. Metric Tree and Variance Decomposition (PVM)

Deconstruct the metric anomaly into deterministic mathematical components.

**Map the KPI driver tree.** Every node must multiply or add to its parent exactly:

- Revenue tree: `Revenue = Traffic x Conversion Rate x Average Order Value`
- Margin tree: `Gross Profit = Volume x (Average Unit Price - Unit COGS) - Direct Fulfillment`

**Execute the Price-Volume-Mix decomposition** across products, channels or regions:

```
Price Effect  = SUM[ Q_actual x (P_actual - P_budget) ]
Volume Effect = (Q_total_actual - Q_total_budget) x P_budget_avg x Mix_budget
Mix Effect    = SUM[ Q_total_actual x (Mix_actual - Mix_budget) x P_budget ]
```

The three effects must reconcile to the total delta. If they do not sum to the reported variance,
the tree is wrong — find the missing node before reporting anything.

Report the exact percentage contribution of Price, Volume and Mix to the top-line delta.

### 2. Causal Attribution and Anomaly Validation

Separate a genuine operational root cause from seasonality or macro noise.

**Baseline vs. counterfactual:**

- Build a control group from unexposed segments, stores or regions.
- Difference-in-Differences: `Delta = (Y_treat,post - Y_treat,pre) - (Y_ctrl,post - Y_ctrl,pre)`
- **Test the parallel trends assumption** on the pre-intervention window. DiD without it is just
  two subtractions wearing a lab coat.

**Statistical significance:**

- 95% confidence intervals and p-values via cluster-robust standard errors or non-parametric
  bootstrapping. Cluster at the level of treatment assignment, not the observation.
- Effect size as Cohen's d or percent change against the synthetic baseline. Significance without
  magnitude tells an executive nothing.

**Rule out confounders** explicitly: promotional calendar shifts, holiday timing, competitor
pricing actions, assortment changes. Name the ones checked and the ones that could not be.

### 3. Stochastic Scenario Modeling and EMV

Replace point estimates with distributions for every strategic option.

**Define parameter distributions** for the uncertain inputs — implementation cost, demand
elasticity, ramp-up time:

- Triangular or Beta-PERT for expert-estimated ranges (Min, Most Likely, Max).
- Log-normal for skewed operational costs and delay durations.

**Run the Monte Carlo** at 5,000-10,000 iterations:

- Outcome distribution: P10 (conservative), P50 (median), P90 (optimistic).
- Expected Monetary Value: `EMV = E[Payoff]`, the probability-weighted outcome.
- Value at Risk (VaR 95%) and Conditional VaR / Expected Shortfall for the tail.

Structure the decision tree over Option A (recommended), Option B (contingency) and Status Quo.
Report the full distribution, never the mean alone.

### 4. Sensitivity and Critical Thresholds (Tornado)

Find the variables that actually govern the outcome.

**One-at-a-time sensitivity:** vary each input between its 10th and 90th percentile with all others
locked at median. Rank by resulting outcome variance — that ranking is the tornado.

**Break-even / switching thresholds:** the exact value at which the recommended option stops
winning. *"If the supplier tariff exceeds 8.2%, Option B becomes optimal."* A threshold is more
actionable than a probability, because someone can watch it.

**Cost of Delay:** `Cost of Delay = Status Quo loss per day + deferred opportunity gain per day`.
This is what makes the decision deadline real rather than arbitrary.

### 5. Executive Synthesis Bridge

Format the technical output into the fields `executive-brief-hook` consumes:

| Brief field | What this skill supplies |
|---|---|
| **Hook data point** | Annualized exposure, VaR 95% downside, or upside arbitrage with confidence bounds |
| **BLUF verdict** | Recommendation by maximum risk-adjusted EMV and shortest payback |
| **Driver: Delta** | Verified PVM variance — *"-$320K, 72% driven by mix shift to low-margin SKUs"* |
| **Driver: Root cause** | Causal driver isolated against the control group |
| **Driver: Business toll** | Annualized run-rate impact and runway effect |
| **Option 1** | Cost, timeline, P50 net gain, downside mitigation |
| **Option 2** | Lower cost/risk profile, conservative upside |
| **Status Quo** | Monetary erosion derived from Cost of Delay |
| **Decision deadline** | Hard operational cutoff: inventory runway, contract expiry, sprint lead time |

Every figure handed to the brief carries its uncertainty band and the one assumption that would
overturn it.

## Gotchas

- **Conflating correlation with causation.** A conversion drop coinciding with a feature release
  proves nothing without a counterfactual. Build the control group or label the finding as
  correlational.
- **Point estimates without bounds.** "$400K savings" misleads; "$280K-$520K at 90% confidence"
  informs. An executive calibrates risk from the width of the interval.
- **Ignoring the Cost of Delay.** Waiting for incremental statistical precision routinely costs more
  than the variance being resolved. Compute the daily cost of indecision before recommending more study.
- **Omitting cannibalization.** Upside for a new line or promo must subtract the margin lost on the
  substitute category it displaces.
- **Fabricating a monetary hook.** When the source data has no revenue or margin figure, the honest
  hook is a verified operational count. Inventing a currency figure to satisfy the brief template
  destroys the credibility of everything else on the page.
- **Over-complicating the hand-off.** Simulation scripts, test statistics and model diagnostics
  belong in a technical appendix. The brief gets decision-ready metrics only.
