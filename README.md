# Market Basket Analysis — Apriori from Scratch

A data mining project that implements the **Apriori algorithm from scratch** to discover frequent itemsets and association rules from retail transaction data, validated against the `mlxtend` library.

## Overview

| | |
|---|---|
| **Dataset** | ShopCT.csv (940 transactions, 11 items) |
| **Algorithm** | Apriori (from scratch) |
| **Min Support** | 15% |
| **Min Confidence** | 60% |
| **Frequent Itemsets Found** | 17 (L1: 11, L2: 5, L3: 1) |
| **Association Rules Found** | 3 |

## Results

### Frequent Itemsets (Top)

| Itemset | Support |
|---|---|
| {frozenmeal, beer, cannedveg} | 15.53% |
| {frozenmeal, beer} | 18.09% |
| {frozenmeal, cannedveg} | 18.40% |

### Association Rules

| Antecedent | Consequent | Support | Confidence | Lift |
|---|---|---|---|---|
| beer, cannedveg | frozenmeal | 15.53% | 87.43% | 2.72 |
| beer, frozenmeal | cannedveg | 15.53% | 85.88% | 2.66 |
| cannedveg, frozenmeal | beer | 15.53% | 84.39% | 2.71 |

> Results match mlxtend exactly — validating the correctness of the from-scratch implementation.

## Output Graphs

- `frequent_itemsets.png` — Support of all frequent itemsets
- `association_rules.png` — Top rules by confidence and lift
- `comparison.png` — Side-by-side comparison with mlxtend

## Requirements

```bash
pip install pandas matplotlib mlxtend
```

## Run

```bash
python main.py
```
