# Retail Metrics Reference

Commercial definitions and unit economics. Each metric lists its formula, the grain it is valid at,
and the way it is most commonly computed wrong.

---

## Transaction-level metrics

### Average Order Value (AOV)

```
AOV = Net Revenue / Number of Orders
```

- **Grain:** order (not line item). Deduplicate to distinct order IDs first.
- **Common error:** dividing by line count instead of order count, which reports average line value.
- Net of returns and discounts, consistent with the stated monetary convention. Report the median
  alongside the mean — AOV distributions are right-skewed and the mean overstates the typical basket.

### Units Per Transaction (UPT)

```
UPT = Total Units Sold / Number of Transactions
```

Measures basket breadth. Rising AOV with flat UPT is price/mix driven; rising AOV with rising UPT is
genuine basket building, which is what cross-merchandising is supposed to produce.

### Average Unit Retail (AUR)

```
AUR = Net Revenue / Units Sold
AOV = UPT * AUR
```

That identity is the cleanest AOV decomposition: any AOV movement is either more items or pricier
items. Always decompose before explaining an AOV change.

### Items per basket vs. categories per basket

Track both. UPT rising within one category is depth; categories per basket rising is the
cross-sell objective and the metric association rules are supposed to move.

---

## Margin and profitability

### Gross Margin

```
Gross Margin $ = Net Revenue - COGS
Gross Margin % = Gross Margin $ / Net Revenue
Markup %       = Gross Margin $ / COGS
```

Margin % and markup % are different numbers and are constantly confused. A 50% margin is a 100%
markup.

### Contribution Margin

```
Contribution Margin = Net Revenue - COGS - Variable Selling Costs
```

Variable selling costs in retail: payment processing, picking/packing, last-mile shipping, returns
handling, and markdown allowance. For e-commerce, shipping and returns often flip a positive gross
margin negative — this is the number that should drive delisting decisions, not gross margin.

### GMROI (Gross Margin Return on Inventory Investment)

```
GMROI = Gross Margin $ / Average Inventory Cost
```

where `Average Inventory Cost` is at cost, not retail. Equivalent decomposition:

```
GMROI = Gross Margin % / (1 - Gross Margin %) * Inventory Turnover      [approx., cost basis]
GMROI = Margin % * Turnover / (Cost %)
```

- **Above 1.0** means the SKU returns more gross margin than the cash tied up in it.
- **Grain:** SKU, category or department over a stated period (usually annual, or period-annualized).
- **Common error:** mixing retail-valued inventory with cost-valued margin. Pick cost and stay there.
- GMROI is the single best one-number assortment screen because it fuses margin and velocity: a
  high-margin slow mover and a low-margin fast mover can land in the same place, correctly.

### GMROF / GMROS (per square foot / per selling space)

```
GMROS = Gross Margin $ / Selling Area
```

Use for physical-store assortment and planogram decisions, where shelf space is the scarce resource
rather than cash.

---

## Inventory metrics

### Inventory Turnover

```
Turnover = COGS / Average Inventory at Cost
Days of Supply (DOS) = 365 / Turnover
```

Average inventory should be the mean of period-end balances across the period, not (begin + end)/2,
which is badly biased for seasonal businesses.

### Sell-Through Rate

```
Sell-Through % = Units Sold / (Units Sold + Units On Hand)
```

or, for a buy-based view:

```
Sell-Through % = Units Sold / Units Received
```

- **Grain:** SKU x season, or SKU x store x week.
- State which denominator you used — they answer different questions (current position vs. buy
  quality) and differ sharply when there are mid-season replenishments.
- Benchmark against the elapsed share of the season. 40% sell-through at week 4 of 12 is ahead of
  plan; the raw number alone says nothing.

### Weeks of Supply (WOS)

```
WOS = Current On-Hand Units / Average Weekly Unit Sales
```

Use a trailing 4–8 week average and exclude stockout weeks from the denominator, or WOS will look
comfortable precisely on the SKUs that keep running out.

### Stockout / In-Stock Rate

```
In-Stock % = Store-SKU-Days with on-hand > 0 / Total Store-SKU-Days
```

Measure at the store-SKU-day grain. Chain-level in-stock percentages hide the concentrated
availability failures that actually cost sales.

### Shrink

```
Shrink % = (Book Inventory - Physical Inventory) at retail / Net Revenue
```

---

## Customer metrics

### Recency, Frequency, Monetary

Defined in `statistical-methods.md` §5. Scoring convention: quintiles 1–5 per dimension, with
recency reversed (5 = most recent). Use quantile cuts computed on the analysis population, and
record the cut points — RFM scores are not comparable across refreshes otherwise.

### Repeat Rate and Retention

```
Repeat Rate   = Customers with >= 2 orders / Total customers
Retention (cohort c, period t) = Active customers from cohort c in period t / Size of cohort c
```

Retention must be computed on a cohort, not on the whole base. Base-level "retention" moves with
acquisition volume and tells you nothing.

### Customer Lifetime Value

```
Historical CLV = Σ (Gross Margin per order)                     over observed history
Predicted CLV  = E[transactions] * E[margin per transaction] * discount factor
```

Use **margin**, not revenue. Revenue-based CLV systematically overvalues low-margin, high-frequency
customers. Discount at the firm's cost of capital when the horizon exceeds a year.

### Purchase Frequency and Inter-Purchase Interval

```
Purchase Frequency = Orders / Distinct Customers          (per period)
IPI = mean(days between consecutive orders)               (per customer, needs >= 2 orders)
```

The median IPI is the right basis for a churn threshold (see the Kaplan-Meier note): a customer
silent for `2 x median IPI` is the standard working definition.

---

## Pricing and promotion metrics

### Discount Depth and Promo Penetration

```
Discount Depth %    = (List Price - Actual Price) / List Price
Promo Penetration % = Units sold on promotion / Total units sold
```

### Promotional Lift and Incrementality

```
Lift %          = (Promo Period Units - Baseline Units) / Baseline Units
Incremental $   = (Promo Units - Baseline Units) * Promo Margin per Unit
ROI             = (Incremental Margin - Promo Cost) / Promo Cost
```

Incremental margin must be net of:
- **Cannibalization** — volume stolen from substitute SKUs in the same category.
- **Pull-forward** — volume that would have occurred in the following weeks anyway. Check the
  post-promotion trough; a deep one means much of the "lift" was timing, not demand.
- **Subsidy** — units that would have sold at full price to customers who would have bought anyway.

A promotion with positive gross lift and negative net incrementality is the normal case, not the
exception. Report net.

### Markdown metrics

```
Markdown %       = Markdown $ / Net Revenue
Realized Margin  = (Net Revenue - COGS) / Net Revenue        after all markdowns
```

Optimal markdown timing follows from the elasticity estimate and the inventory decay: mark down when
the expected margin from holding (sell-through at current price x remaining weeks) falls below the
expected margin from clearing now.

---

## Store and channel metrics

### Sales per Square Foot / Metre

```
Sales per Area = Net Revenue / Selling Area
```

### Conversion Rate

```
Conversion = Transactions / Footfall              (physical)
Conversion = Orders / Sessions                    (digital)
```

Digital conversion must fix the denominator's definition (sessions vs. users vs. visits) and keep it
fixed. Most conversion "improvements" are denominator changes.

### Like-for-Like / Comparable Sales

```
LFL Growth % = (Revenue this period - Revenue same period last year) / Revenue same period last year
```

restricted to stores open and trading in both periods, with no major refit. The comparable-store
filter is the whole metric; without it this is just total growth.

---

## The metric tree

Every executive retail number decomposes to the same few drivers. Use this to route a finding to an
action:

```
Revenue
├── Traffic (footfall / sessions)
├── Conversion
└── AOV
    ├── UPT  ──── driven by assortment, adjacency, cross-sell  → association rules
    └── AUR  ──── driven by price, mix, discount depth         → elasticity, markdown

Gross Margin $ = Revenue x Margin %
GMROI          = Gross Margin $ / Average Inventory at Cost
                 ├── Margin %   → pricing, sourcing, markdown discipline
                 └── Turnover   → ABC-XYZ, forecast accuracy, availability
```

When reporting, always land the statistical finding on one of these nodes. "Lift of 1.8 between
category A and B" is not a finding; "moving B adjacent to A is a UPT lever worth an estimated X" is.
