"""
Run the wheelchair fit logic in batch (Node, no browser). Append fitted dimensions to a dataframe.

Requires: Node.js and wheelchair-fit-node/ (run npm install there). Model files live in that directory.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _wc_type_to_tool(wc_type: str) -> str:
    """Map volunteer WC Type to tool wheelchair type (manual / powered). Stroller -> powered."""
    if pd.isna(wc_type):
        return "manual"
    s = str(wc_type).strip().lower()
    if s in ("power", "powered", "stroller"):
        return "powered"
    return "manual"


def run_fit_batch(
    df: pd.DataFrame,
    *,
    id_col: str = "ID",
    gender_col: str = "Gender",
    age_col: str = "Age",
    height_col: str = "Height (cm)",
    bmi_col: str = "BMI",
    wc_type_col: str = "WC Type",
    script_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    For each row in df with valid demographics, run the fit and return fitted seat dimensions.

    Rows with NaN or empty in any of ID, Gender, Age, Height (cm), BMI, or WC Type are skipped.
    Returns a DataFrame with columns: id_col, seatWidth, seatDepth, seatPanHeight (inches).
    """
    root = Path(__file__).resolve().parent
    if script_path is None:
        script_path = root / "wheelchair-fit-node" / "fitHeadless.mjs"
    script_path = Path(script_path)
    if not script_path.is_file():
        raise FileNotFoundError(f"Fit script not found: {script_path}")

    required = [id_col, gender_col, age_col, height_col, bmi_col, wc_type_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing columns required for fit: {missing}")
    valid_df = df.dropna(subset=required, how="any")

    if len(valid_df) == 0:
        return pd.DataFrame(columns=[id_col, "seatWidth", "seatDepth", "seatPanHeight"])

    rows = []
    for _, r in valid_df.iterrows():
        rows.append({
            "id": r[id_col],
            "gender": r[gender_col],
            "age": r[age_col],
            "heightCm": r[height_col],
            "bmi": r[bmi_col],
            "wheelchairType": _wc_type_to_tool(r.get(wc_type_col, "manual")),
        })

    input_json = json.dumps(rows)
    result = subprocess.run(
        ["node", str(script_path)],
        input=input_json,
        capture_output=True,
        text=True,
        timeout=60 + len(rows) * 5,
        cwd=str(script_path.parent),
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Fit batch failed (exit {result.returncode}): {result.stderr or result.stdout}"
        )

    out = json.loads(result.stdout)
    return pd.DataFrame(out).rename(columns={"id": id_col})


if __name__ == "__main__":
    # Example: load joined CSV and run fit batch
    root = Path(__file__).resolve().parent
    joined_path = root / "volunteer_with_digitizations.csv"
    if not joined_path.exists():
        print("Run main.py first to create volunteer_with_digitizations.csv", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(joined_path)
    fitted = run_fit_batch(df)
    out_path = root / "volunteer_fitted.csv"
    fitted.to_csv(out_path, index=False)
    print(f"Wrote {len(fitted)} rows to {out_path}")
