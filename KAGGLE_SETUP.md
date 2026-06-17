# Getting the Kaggle Dataset — Step by Step

This adds **living area sqft** to your dataset, the single biggest missing
feature. Follow these steps exactly in order.

---

## Step 1 — Download from Kaggle

### Option A: Browser (no setup, recommended if you haven't used Kaggle API before)

1. Open this URL in your browser:
   ```
   https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset
   ```

2. If you don't have a Kaggle account, click **Register** (top right) — it's free, takes 1 minute.

3. Click the blue **Download** button (top right of the dataset page).

4. You'll download a file called **`realtor-data.zip.csv`** (about 90 MB).

5. **Move that file into your `data/` folder:**
   ```
   housing_project/
   └── data/
       ├── cleaned_dataset.xlsx        ← already there
       └── realtor-data.zip.csv        ← put it here
   ```

### Option B: Kaggle API (faster if you do this often)

```bash
# Install the CLI
pip install kaggle

# Get your API token:
# Kaggle → top-right avatar → Settings → API → "Create New Token"
# This downloads kaggle.json — save it to ~/.kaggle/kaggle.json

# Download + unzip directly into the data folder
cd housing_project/data
kaggle datasets download -d ahmedshahriarsakib/usa-real-estate-dataset --unzip
```

The file will be named `realtor-data.zip.csv` after unzipping.

---

## Step 2 — Run the enrichment script

```bash
cd housing_project
python src/enrich_with_kaggle.py
```

This takes about 1–2 minutes. You'll see output like:

```
[enrich] Loading Kaggle dataset from data/realtor-data.zip.csv ...
[enrich] Raw Kaggle rows: 2,226,382   cols: [...]
[enrich] After filtering to 16 states: 842,150 rows
[enrich] After cleaning: 798,421 usable rows
[enrich] Computing city-level sqft statistics ...
[enrich] City-level stats: 3,847 (state, city) combinations
[enrich] City-level match: 6,412 / 6,795 rows (94.4%)
[enrich] State-level fallback applied to 383 rows

[enrich] === Enrichment validation ===
Correlations with price_value:
  median_ppsqft          +0.47  ██████████████
  median_sqft            +0.38  ███████████
  p75_sqft               +0.37  ███████████
  ...
  beds                   +0.20  ██████

[enrich] Saved enriched dataset -> data/cleaned_dataset_enriched.xlsx
```

If you see `City-level match: 94%+` you're in great shape.
If match rate is below 80%, check that the file is named correctly (Step 1).

---

## Step 3 — Run the full pipeline

```bash
python src/pipeline.py
```

`preprocess.py` automatically detects `cleaned_dataset_enriched.xlsx` and
uses it. No config changes needed.

---

## What happens under the hood

The script does NOT replace your dataset with Kaggle data. Instead it:

1. Loads the 2.2M Kaggle listings, filters to your 16 states (~800k rows)
2. Computes **per-city statistics**: median sqft, p25/p75 sqft, median price-per-sqft
3. Joins those stats to your 6.8k rows on `city + state`
4. Cities not in Kaggle (~5-6% of your rows) get the **state-level median** as fallback

This way you keep all your existing NLP features (sentiment, luxury signals,
amenities from listing remarks) AND gain the sqft signal. Best of both datasets.

---

## Expected results after retraining

| Metric | Before (no sqft) | After (with Kaggle sqft) |
|---|---|---|
| Best test R² | 0.611 | ~0.72–0.78 |
| Best overfit gap | 0.351 | ~0.10–0.15 |
| Tier classifier accuracy | 75.6% | ~80–84% |

---

## Troubleshooting

**"Kaggle CSV not found"** — make sure the file is in `data/` and named
`realtor-data.zip.csv`. If Kaggle gave it a different name, pass it explicitly:
```bash
python src/enrich_with_kaggle.py --kaggle data/YOUR_FILENAME.csv
```

**Low city match rate (<70%)** — city name casing differences. The script
normalises both sides to lowercase so this is rare, but check
`data/kaggle_city_stats.csv` to see which cities were found.

**"ModuleNotFoundError: openpyxl"** — run `pip install openpyxl`.
