# Wheelchair Fit Integration (Data Analysis)

This project is a **data analysis pipeline**: it joins volunteer metadata with digitization measurements, runs the wheelchair fit logic (same math as the [web tool](https://humanshape.org/WheelchairTool/)) on each row’s demographics, and writes one CSV with both measured and fitted dimensions.

No browser or web server is required. Fit runs in Node using the same formulas as the tool.

## How it works

1. **main.py** loads Excel volunteer data and digitization data, merges them, then calls the fit for each row (Gender, Age, Height, BMI, WC Type).
2. **wheelchair_fit.run_fit_batch** keeps only rows with valid demographics (no NaN/empty in ID, Gender, Age, Height, BMI, WC Type), builds input JSON, and runs **wheelchair-fit-node/fitHeadless.mjs** (Node, no browser).
3. **fitHeadless.mjs** loads the mean PLY model and PCA data from `wheelchair-fit-node/model/`, morphs the mesh per row, and computes optimal seat width, depth, and pan height (same logic as the web app).
4. Fitted columns (`seatWidth_fit`, `seatDepth_fit`, `seatPanHeight_fit`) are merged back into the joined dataframe by ID; rows that were skipped (invalid demographics) get NaN for those columns. Output is **volunteer_with_digitizations.csv**.

## Setup

1. **Headless fit (Node)** in this repo:
   ```bash
   cd wheelchair-fit-node && npm install
   ```
   Requires **model/mean_model_tri.ply** and **model/Anth2Data.csv** in `wheelchair-fit-node/model/` (included in the extracted fit component).

2. **Python**: pandas and openpyxl (for Excel). No Playwright.

## Usage

Run the full pipeline (digitizations + fit + export):

```bash
python main.py
```

Output: **volunteer_with_digitizations.csv** with volunteer metadata, digitization columns (e.g. seatWidth, seatDepth, panHeight), and fitted columns (seatWidth_fit, seatDepth_fit, seatPanHeight_fit).

## Input/output

- **Input** (per row): ID, Gender, Age, Height (cm), BMI, WC Type (Manual/Power/Stroller → manual/powered).
- **Fitted columns**: seatWidth_fit, seatDepth_fit, seatPanHeight_fit (inches), from the same formulas as the web tool.

You can compare these fitted values with the digitization-derived seatWidth, seatDepth, panHeight in the same CSV.
