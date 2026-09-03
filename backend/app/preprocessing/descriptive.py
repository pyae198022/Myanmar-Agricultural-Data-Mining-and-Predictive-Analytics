"""
Descriptive Mining computations for the Data Statistics / Descriptive Mining pages.

This module implements the four descriptive-mining methods from the Project Book,
computing every result directly from the REAL bundled cleaned dataset
(``backend/data/cleaned_data.csv``) — nothing is hardcoded or fabricated:

* Correlation Analysis     - Pearson correlation matrix of the numeric columns.
* Association Rule Mining  - Apriori rules for Yield_Level and Crop_Type.
* Frequent Pattern Mining  - Apriori frequent itemsets for Crop Yield and Crop Type.
* Sequential Pattern Mining- temporal order-2 / order-3 patterns by (Region, Crop_Type).

It reuses the project's existing data-loading utility (``reference.load_clean_data``)
and its canonical ``YIELD_LEVEL_THRESHOLD`` so results stay consistent with the
trained models. It never modifies the dataset or any model artifact.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from . import reference

# Mining constants from the Project Book (reproduced, not hardcoded into the UI).
MIN_SUPPORT_YIELD = 0.05    # Crop Yield (incl. Yield_Level) itemset support
MIN_SUPPORT_CROP = 0.03     # Crop Type needs a lower support to yield itemsets
MAX_ITEMSET_SIZE = 3
MIN_CONFIDENCE = 0.60       # Yield_Level association rule confidence floor
MIN_CONFIDENCE_CROP = 0.50  # Crop_Type (Project Book: 50%) confidence floor
SUPPORT_SEQUENTIAL = 0.10   # sequential-pattern minimum support (report only)

# Transaction attributes used per Project Book.
YIELD_MINING_COLS = [
    "Region", "Crop_Type", "Soil_Type", "Seeding_Season", "Water_Source", "Yield_Level",
]
CROP_MINING_COLS = [
    "Region", "Soil_Type", "Seeding_Season", "Water_Source", "Crop_Type",
]


# ─── Data loading + labeling ────────────────────────────────────────────────

def _load_with_level() -> pd.DataFrame:
    """Return the cleaned dataset with a derived Yield_Level (Low/High)."""
    df = reference.load_clean_data().copy()
    df["Yield_Level"] = np.where(
        df["Crop_Yield"] > reference.YIELD_LEVEL_THRESHOLD, "High", "Low"
    )
    return df


def _transactions(df: pd.DataFrame, cols: list[str]) -> list[frozenset]:
    """Build frozenset transactions ``col=value`` for the given columns."""
    return (
        df[cols]
        .astype(str)
        .apply(lambda r: frozenset(f"{c}={v}" for c, v in zip(cols, r)), axis=1)
        .tolist()
    )


# ─── Apriori (manual, reproduces the project algorithm) ────────────────────

def _apriori(transactions: list[frozenset], min_support: float, max_k: int = MAX_ITEMSET_SIZE):
    """
    Manual Apriori over transactions. Returns the frequent itemsets dict
    (frozenset -> count) and a list of candidate counts per itemset size.
    """
    n = len(transactions)

    all_items: set[frozenset] = set()
    for t in transactions:
        all_items |= t

    counter_1 = Counter()
    for t in transactions:
        for item in t:
            counter_1[frozenset([item])] += 1

    frequent_1 = {k: v for k, v in counter_1.items() if v / n >= min_support}

    candidate_counts = {1: len(all_items)}

    all_frequent = dict(frequent_1)
    current_level = frequent_1
    k = 2

    while current_level and k <= max_k:
        previous = list(current_level.keys())
        candidates: set[frozenset] = set()
        for i in range(len(previous)):
            for j in range(i + 1, len(previous)):
                union = previous[i] | previous[j]
                if len(union) == k:
                    candidates.add(union)
        candidate_counts[k] = len(candidates)

        counter_k = Counter()
        for t in transactions:
            for cand in candidates:
                if cand.issubset(t):
                    counter_k[cand] += 1

        frequent_k = {k: v for k, v in counter_k.items() if v / n >= min_support}
        all_frequent.update(frequent_k)
        current_level = frequent_k
        k += 1

    return all_frequent, candidate_counts, n


def _itemset_label(item: str) -> str:
    """Human label for a ``col=value`` mining item."""
    if "=" not in item:
        return item
    col, value = item.split("=", 1)
    # Keep the column context clear (e.g. "Region: Sagaing").
    return f"{col.replace('_', ' ')}: {value}"


# ─── Correlation Analysis ───────────────────────────────────────────────────

def correlation_analysis(df: pd.DataFrame) -> dict:
    """Pearson correlation matrix over the numeric columns of the dataset."""
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    mat = df[num_cols].corr(method="pearson")

    matrix = [
        {
            "feature": col,
            "values": [
                {"other": other, "r": float(mat.loc[col, other])} for other in num_cols
            ],
        }
        for col in num_cols
    ]

    return {
        "method": "Pearson correlation coefficient",
        "columns": num_cols,
        "matrix": matrix,
    }


# ─── Frequent Pattern Mining ────────────────────────────────────────────────

def _frequent_itemsets_report(transactions: list[frozenset], min_support: float, label: str) -> dict:
    freq, cand_counts, n = _apriori(transactions, min_support)

    by_size = {1: [], 2: [], 3: []}
    for itemset, count in freq.items():
        size = len(itemset)
        if size in by_size:
            by_size[size].append(
                {
                    "itemset": sorted(_itemset_label(x) for x in itemset),
                    "count": int(count),
                    "support": round(count / n, 4),
                }
            )
    for size_list in by_size.values():
        size_list.sort(key=lambda r: -r["count"])

    return {
        "analysis": label,
        "min_support": min_support,
        "n_transactions": n,
        "candidate_counts": {
            str(k): int(v) for k, v in sorted(cand_counts.items())
        },
        "frequent_counts": {
            str(k): len(by_size[k]) for k in by_size
        },
        "total_candidates": int(sum(cand_counts.values())),
        "total_frequent": int(len(freq)),
        "by_size": {str(k): by_size[k] for k in by_size},
    }


def frequent_patterns(df: pd.DataFrame) -> dict:
    """Frequent itemsets for Crop Yield (5% support) and Crop Type (3% support)."""
    yield_tx = _transactions(df, YIELD_MINING_COLS)
    crop_tx = _transactions(df, CROP_MINING_COLS)

    yield_report = _frequent_itemsets_report(yield_tx, MIN_SUPPORT_YIELD, "Crop Yield")
    crop_report = _frequent_itemsets_report(crop_tx, MIN_SUPPORT_CROP, "Crop Type")

    return {
        "method": "Apriori frequent pattern mining",
        "crop_yield": yield_report,
        "crop_type": crop_report,
    }


# ─── Association Rule Mining ────────────────────────────────────────────────

def _generate_rules(freq: dict, n: int, min_confidence: float) -> list[dict]:
    import itertools

    rules = []
    for itemset, count in freq.items():
        if len(itemset) < 2:
            continue
        itemset_support = count / n
        items = list(itemset)
        for r in range(1, len(items)):
            for antecedent_tuple in itertools.combinations(items, r):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent
                antecedent_count = freq.get(antecedent, 0)
                if antecedent_count == 0:
                    continue
                antecedent_support = antecedent_count / n
                confidence = itemset_support / antecedent_support
                if confidence < min_confidence:
                    continue
                consequent_count = freq.get(consequent, 0)
                consequent_support = consequent_count / n
                if consequent_support == 0:
                    continue
                lift = confidence / consequent_support
                rules.append(
                    {
                        "antecedent": sorted(_itemset_label(x) for x in antecedent),
                        "consequent": sorted(_itemset_label(x) for x in consequent),
                        "support": round(itemset_support, 4),
                        "confidence": round(confidence, 4),
                        "lift": round(lift, 4),
                    }
                )
    rules.sort(key=lambda r: (-r["lift"], -r["confidence"]))
    return rules


def association_rules(df: pd.DataFrame) -> dict:
    """
    Yield_Level association rules (5% support, >=60% confidence, Lift>1) and
    Crop_Type association rules (3% support, >=50% confidence per Project Book),
    computed from the dataset via the same Apriori used to train the advanced
    models.
    """
    yield_tx = _transactions(df, YIELD_MINING_COLS)
    y_freq, _, n = _apriori(yield_tx, MIN_SUPPORT_YIELD)
    yield_rules = _generate_rules(y_freq, n, MIN_CONFIDENCE)
    yield_rules = [
        r for r in yield_rules
        if any(("yield level: " in c.lower()) for c in r["consequent"]) and r["lift"] > 1
    ]

    crop_tx = _transactions(df, CROP_MINING_COLS)
    c_freq, _, n_crop = _apriori(crop_tx, MIN_SUPPORT_CROP)
    crop_rules = _generate_rules(c_freq, n_crop, MIN_CONFIDENCE_CROP)
    crop_rules = [
        r for r in crop_rules
        if any(("crop type: " in c.lower()) for c in r["consequent"])
    ]

    return {
        "method": "Apriori association rule mining",
        "min_confidence": MIN_CONFIDENCE,
        "yield_level": {
            "min_support": MIN_SUPPORT_YIELD,
            "min_confidence": MIN_CONFIDENCE,
            "rule_count": len(yield_rules),
            "rules": yield_rules,
        },
        "crop_type": {
            "min_support": MIN_SUPPORT_CROP,
            "min_confidence": MIN_CONFIDENCE_CROP,
            "rule_count": len(crop_rules),
            "rules": crop_rules,
        },
    }


# ─── Sequential Pattern Mining ──────────────────────────────────────────────

def sequential_patterns(df: pd.DataFrame) -> dict:
    """
    Temporal sequential patterns by (Region, Crop_Type) ordered by Year (2012-2023).
    Duplicate (Region, Crop_Type, Year) records are averaged before level labelling,
    giving the real sequence count (481). Order-2 and order-3 pattern supports are
    computed from the actual dataset. Supports are reported as the share of the
    total number of order-k patterns observed across all sequences.
    """
    agg = (
        df.groupby(["Region", "Crop_Type", "Year"], as_index=False)
        .agg(Crop_Yield=("Crop_Yield", "mean"))
    )
    agg["Yield_Level"] = np.where(
        agg["Crop_Yield"] > reference.YIELD_LEVEL_THRESHOLD, "High", "Low"
    )
    agg = agg.sort_values(["Region", "Crop_Type", "Year"])

    order2: list[tuple[str, str]] = []
    order3: list[tuple[str, str, str]] = []
    for _, g in agg.groupby(["Region", "Crop_Type"]):
        lv = g["Yield_Level"].tolist()
        order2 += [(lv[i], lv[i + 1]) for i in range(len(lv) - 1)]
        order3 += [(lv[i], lv[i + 1], lv[i + 2]) for i in range(len(lv) - 2)]

    def _support_counts(pairs, size):
        counts = Counter(pairs)
        total = len(pairs)
        rows = [
            {
                "sequence": list(k),
                "count": int(c),
                "support": round(c / total, 4) if total else 0,
            }
            for k, c in sorted(counts.items(), key=lambda x: -x[1])
        ]
        # Project Book: report only patterns meeting the minimum support (10%).
        return [r for r in rows if r["support"] >= SUPPORT_SEQUENTIAL]

    order2_report = _support_counts(order2, 2)
    order3_report = _support_counts(order3, 3)

    return {
        "method": "Sequential pattern mining (by Region-Crop_Type over years)",
        "min_support": SUPPORT_SEQUENTIAL,
        "num_sequences": int(agg[["Region", "Crop_Type"]].drop_duplicates().shape[0]),
        "order_2": {
            "total_patterns": len(order2),
            "patterns": order2_report,
        },
        "order_3": {
            "total_patterns": len(order3),
            "patterns": order3_report,
        },
        "note": (
            "Sequence order represents temporal occurrence: consecutive years for "
            "order-2, and three consecutive years for order-3 patterns."
        ),
    }


# ─── Full report ────────────────────────────────────────────────────────────

def descriptive_mining_report() -> dict:
    """Compute the complete descriptive-mining report from the real dataset."""
    df = _load_with_level()

    return {
        "methodology": [
            {
                "id": "correlation",
                "name": "Correlation Analysis",
                "available": True,
                "note": "Pearson correlation of the numeric agricultural attributes.",
            },
            {
                "id": "association",
                "name": "Association Rule Mining",
                "available": True,
                "note": "Apriori rules for Yield_Level and Crop_Type from real data.",
            },
            {
                "id": "frequent_pattern",
                "name": "Frequent Pattern Mining",
                "available": True,
                "note": "Apriori frequent itemsets for Crop Yield and Crop Type.",
            },
            {
                "id": "sequential_pattern",
                "name": "Sequential Pattern Mining",
                "available": True,
                "note": "Order-2 and order-3 temporal patterns by Region-Crop_Type.",
            },
        ],
        "correlation": correlation_analysis(df),
        "association_rules": association_rules(df),
        "frequent_patterns": frequent_patterns(df),
        "sequential_patterns": sequential_patterns(df),
    }
