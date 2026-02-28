# CMBS Data Pipeline — California Retail Properties

**Author:** Ruoling (Iris) Li  
**Affiliation:** UC Irvine, Mathematics & Business Economics  

---

This repository contains the data preparation and analysis work I did on Trepp CMBS loan-level data for California retail properties as part of an ongoing research project on commercial real estate credit risk, supervised by Professor Jack Liebersohn at UC Irvine.

The main contribution here is the cleaning pipeline. The ML notebook is included to show what the cleaned data enables, but the emphasis is on getting the data right first.

---

## Files

**`CMBS_Data_Cleaning.ipynb`** — the main notebook. Walks through the full process of taking raw Trepp data from unusable to analysis-ready: column triage, type coercion, address normalization and geocode merging, feature engineering, and memory optimization. There's a short validation section at the end that runs a quick cross-validated model to confirm the engineered features carry real signal before handing off.

**`ML_Analysis_V5.ipynb`** — downstream modeling on the cleaned output. Covers regression (predicting continuous DSCR) and classification (flagging at-risk loans with DSCR < 1.2) across five model types, with explicit leakage auditing and property-level train/val/test splitting. Included for completeness, but the cleaning notebook is the better starting point.

---

## Data

The underlying data is proprietary Trepp CMBS loan-level data and can't be shared publicly. There're two main dataset we used:

- `data/prop_CA_2019_2021.csv` — raw Trepp loan records
- `data/unique_retail_addresses_arcgis.csv` — geocoded addresses from ArcGI

The cleaning notebook outputs `data/cmbs_clean.csv`, which is what the ML notebook reads in.

---

## Cleaning Pipeline (notebook walkthrough)

**Column triage.** The raw file has hundreds of columns, many of them nearly empty. The first step audits missingness at several thresholds and drops anything below 50% non-null. The counts are printed before dropping so the decision is visible and easy to adjust.

**Type coercion.** Everything comes in as strings. Dates are parsed with `errors='coerce'` so malformed entries become `NaT` rather than crashing or, worse, silently producing wrong values. Same approach for numerics. The number of newly created nulls is printed per column so unexpected parsing failures surface immediately.

**Address normalization and geocode merge.** Raw addresses are inconsistently formatted — mixed casing, varying abbreviations, extra punctuation. A normalizer function standardizes everything to uppercase with consistent abbreviations (STREET→ST, AVENUE→AVE, etc.) before building a city/state/zip-aware join key. Coordinates from an ArcGIS geocoding export are then merged in, keeping only high-confidence matches (Score ≥ 85).

**Feature engineering.** Income and expense figures are normalized per square foot to make properties comparable across size. Tenant lease data (top 3 tenants) is used to build rollover exposure metrics — what percentage of rent is expiring in the next 12 months, weighted average unexpired lease term, HHI concentration. All division operations use a `safe_div()` helper that returns NaN rather than infinity when the denominator is zero. Percent columns that arrive on a 0–100 scale are auto-detected and rescaled. PSF features are winsorized at the 1st/99th percentile.

**Memory optimization.** A documented `optimize_dtypes()` function brings the DataFrame down to a manageable size: dropping high-cardinality ID/string columns, converting datetimes to integer days, downcasting float64→float32 and int64→int32, converting low-cardinality strings to category. Typically cuts size by around 50%.

**Validation.** A final completeness table prints non-null counts for all key features. The short ML sanity check that follows (Logistic Regression and Random Forest, 5-fold CV) is there purely to confirm the cleaned features carry predictive signal — if ROC-AUC is near 0.5, something upstream is wrong.

---

## Requirements

```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
matplotlib>=3.6
seaborn>=0.12
```

```bash
pip install -r requirements.txt
```

---

## Reproducibility

All paths are relative. Random seeds are set explicitly (`np.random.seed(42)`) before any stochastic operations. Key thresholds — missingness cutoff, LTV flag, occupancy flag, geocode score minimum — are defined as named constants at the point of use rather than buried in logic, so they're easy to find and adjust.
