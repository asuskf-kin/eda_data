# Gotchas

The five failure modes that produce confident, wrong retail analysis. Each one below gives the
mechanism, a detection check you can run, and the fix.

---

## 1. Treating returns as standard negative sales

**Mechanism.** Return lines arrive in the same table as sales with negative quantity and negative
amount. Aggregating them in silently corrupts three things at once:

- `log(x)` and `log1p(x)` are undefined or nonsensical on negatives.
- Variance and CV are inflated by the ± churn, so a stable SKU with a high return rate is classified
  Z (lumpy) when its *demand* is perfectly steady.
- Elasticity regressions get a negative quantity on the left-hand side and either drop the row or
  produce nonsense.

And the business signal is lost: a 30% return rate is a major finding that disappears entirely when
returns net out against sales before anyone looks.

**Detection.**

```python
df.filter(pl.col("cantidad") < 0).height          # how many?
df.filter(pl.col("importe") < 0).height
# Do the signs agree? A negative amount with positive quantity is a different defect (price adj.)
df.filter((pl.col("cantidad") > 0) & (pl.col("importe") < 0)).height
```

Also check for **return-without-sale**: a return line whose SKU/customer has no prior purchase in the
window. Those are either out-of-window originals or data errors, and they break any matched analysis.

**Fix.**

1. Split the frame: `ventas` (positive), `devoluciones` (negative), `ajustes` (anything else —
   price corrections, voids, test records).
2. Run the distributional and elasticity work on `ventas` only.
3. Run a separate returns analysis: return rate by SKU, category, channel, customer and time-to-
   return distribution. Return behaviour is its own finding, often the most actionable one.
4. Reunite only at the P&L level, where net revenue is the correct number.
5. Where a matched return exists, decide explicitly whether the original sale still counts as demand
   (it does for availability and forecasting; it does not for margin).

---

## 2. Endogeneity in price elasticity

**Mechanism.** OLS on `ln(Q) ~ ln(P)` identifies the demand curve only if price moves for reasons
unrelated to demand. In retail it almost never does:

- Prices are **cut** when demand is weak (clearance) → drives the estimate toward zero or positive.
- Prices are **raised** into peak season, when demand is strong (holiday, back-to-school) → the
  fitted line slopes upward.
- Promotions are **scheduled** on items already expected to sell → the promotion "causes" a lift
  that was coming anyway.

The result is the classic supply-demand identification problem: you trace out an equilibrium locus,
not a demand curve. A positive own-price elasticity is the loud version; a merely attenuated
negative one (say -0.3 where the truth is -1.8) is the dangerous version, because it looks
plausible and leads to a price increase that destroys volume.

**Detection.**

- Sign check: `b > 0` on a normal good → specification failure, full stop. Do not rationalize it.
- Correlate price changes with the seasonal component from the STL decomposition. Significant
  correlation means price is moving with demand.
- Compare the estimate on promotional price variation only vs. regular-price variation only. A large
  gap means the two sources are identifying different things.
- Check whether price variation is mostly *clearance* by cross-tabulating price changes with
  end-of-lifecycle or high-inventory flags.

**Fix, in order of preference.**

1. **Experiment.** Randomized or staggered price tests across stores/regions. This is the only clean
   answer; everything else is mitigation.
2. **Fixed effects.** SKU fixed effects (absorb permanent quality/price-level differences) plus time
   fixed effects (absorb common demand shocks). This is the minimum credible specification.
3. **Instruments.** Cost shocks, competitor prices in non-overlapping markets (Hausman instruments),
   or wholesale price changes — anything that shifts price without shifting local demand. Report the
   first-stage F statistic; below ~10 the instrument is weak and the IV estimate is worse than OLS.
4. **Restrict the window.** Estimate on regular-price periods only, and treat promotional response
   as a separate uplift model rather than as points on the same demand curve.

Always report the specification alongside the number. An elasticity without its identifying
assumption is not a result.

---

## 3. Data leakage in customer aggregates

**Mechanism.** RFM, CLV and segment features are computed over "all available data", then used to
predict an outcome that occurred inside that same window. The feature contains the answer.

Typical forms:

- Monetary value computed to the end of the dataset, then used to predict churn "next quarter" —
  but the last quarter's purchases are already inside `M`.
- Recency computed relative to `max(date)` in the whole table rather than the cutoff, so a customer
  who lapsed and came back looks recently active before the return happened.
- Segment labels assigned on full history, then used to explain behaviour that determined the label.

The result is an analysis with excellent in-sample separation and no predictive value, which is
worse than no analysis because it gets acted on.

**Detection.**

```python
t_split = date(2026, 6, 30)
# Every feature must satisfy this:
assert features_source.select(pl.col("fecha").max()).item() <= t_split
```

- Recompute a key feature with a cutoff 30 days earlier. If segment membership shifts dramatically,
  the feature was being driven by the most recent window.
- Look for impossibly strong predictors. A feature with AUC > 0.95 for churn is leakage until
  proven otherwise.

**Fix.**

1. Declare `T_split` once, at the top of the analysis, as a constant.
2. Build all features from `fecha <= T_split`, all outcomes from `fecha > T_split`.
3. Anchor recency to `T_split`, never to `max(fecha)`.
4. Require a minimum observation window per customer (a customer acquired 3 days before `T_split`
   has no meaningful F or M) and either exclude them or model them as a separate "new" segment.
5. For backtesting, roll `T_split` across several dates and check stability rather than tuning on one.

---

## 4. Aggregating without weighting (Simpson's Paradox)

**Mechanism.** The unweighted mean of store-level ratios is not the chain ratio. When stores differ
in size, the small stores get the same vote as the flagships, and the aggregate can move opposite to
every one of its components.

The canonical retail version: every store's conversion rate improves year over year, but chain
conversion falls, because volume shifted toward the lower-converting format. Both statements are
true; only one of them is the answer to "did we get better at converting?" — and which one depends
on the question being asked.

**Detection.**

```python
# Compute both and compare
no_ponderado = df.group_by("tienda").agg(
    (pl.col("ingreso").sum() / pl.col("ordenes").sum()).alias("aov")
)["aov"].mean()

ponderado = df["ingreso"].sum() / df["ordenes"].sum()
```

A material gap between the two is the alarm. Also check whether group sizes correlate with the
metric — that correlation is exactly the condition under which the paradox bites.

**Fix.**

1. Ratios aggregate as **ratio of sums**, never mean of ratios. `AOV_chain = Σrevenue / Σorders`.
2. When a per-store average is genuinely what is wanted, weight by the right exposure: orders,
   footfall, selling area or trading days, and name the weight in the output.
3. Report both figures when they disagree, and explain the mix shift — the disagreement is usually
   the more interesting finding than either number.
4. For like-for-like comparisons, hold the mix fixed explicitly (same store set, same period length,
   same trading-day count) rather than hoping the average absorbs it.

---

## 5. Confusing correlation with association

**Mechanism.** Two bestsellers co-occur in many baskets simply because each appears in many baskets.
If bread is in 40% of baskets and milk in 35%, they will share about 14% of baskets under complete
independence. A support of 14% looks impressive and means nothing.

The mirror error is relying on lift alone: a pair with lift 8.0 built on 12 transactions out of
2 million is statistical noise, and acting on it means rearranging a planogram for nobody.

**Detection.**

For every candidate rule, carry all four numbers:

```
Support(A), Support(B), Support(A∩B), Lift
```

and compare `Support(A∩B)` against the independence baseline `Support(A) * Support(B)`. Then:

- Fisher's exact test on the 2x2 table, with Benjamini-Hochberg across all tested pairs.
- A minimum absolute count (typically ≥ 30–50 co-occurring baskets) before a rule is even eligible.
- Leverage (`Support(A∩B) - Support(A)*Support(B)`) to rank by absolute business volume rather than
  by ratio.

**Fix.**

1. Filter first on support and absolute count, then rank by lift, then test for significance, then
   sanity-check the survivors against merchandising logic.
2. Read lift and support together as a 2x2 map — anchors (high support, low lift) and affinity
   drivers (moderate support, high lift) call for opposite actions. Moving an anchor is a mistake;
   moving an affinity driver next to its partner is the opportunity.
3. Beware **confounded co-occurrence**: two items may co-occur because of a shared third cause —
   the same shopping trip type, a bundle promotion, or a store layout that already places them
   together. Condition on trip type or check whether the lift survives outside promotion periods.
4. Direction matters. `Confidence(A→B) ≠ Confidence(B→A)` even though lift is symmetric. The
   asymmetry tells you which item to place near which.

---

## Cross-cutting: the questions to ask before shipping any retail EDA finding

1. What is the grain, and is the key actually unique at that grain?
2. Are returns separated? Are zeros real zeros or censored stockouts?
3. Is the cutoff date enforced on every customer feature?
4. Is every ratio a ratio of sums?
5. Is the number robust to the top 1% of observations, or is it those observations?
6. Does the estimate rest on variation that was generated by the thing it is trying to measure?
7. Would the recommendation change if the assumption I am least sure about were wrong?
