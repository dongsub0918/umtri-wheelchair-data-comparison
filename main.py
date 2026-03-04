from xlsxReader import XlsxReader
from digitizationsReader import DigitizationsReader
from xyzFileGenerator import run_reconstruct
from wheelchair_fit import run_fit_batch

run_reconstruct("digitizations")

reader = XlsxReader()
reader.load("VolunteerTestSummary20251024.xlsx")

digitizationsReader = DigitizationsReader()
digitizationsReader.load("digitizations", allowed_ids=set(reader.df["ID"]))

joined = reader.df.merge(
    digitizationsReader.df,
    on="ID",
    how="left",
)

# Fit wheelchair to human demographics for each row; append fitted dimensions
fitted = run_fit_batch(joined)
joined = joined.merge(
    fitted[["ID", "seatWidth", "seatDepth", "seatPanHeight"]].rename(
        columns={
            "seatWidth": "seatWidth_fit",
            "seatDepth": "seatDepth_fit",
            "seatPanHeight": "seatPanHeight_fit",
        }
    ),
    on="ID",
    how="left",
)

# Fitted values are in inches -> convert to mm and round to 2 decimals
INCH_TO_MM = 25.4
for col in ("seatWidth_fit", "seatDepth_fit", "seatPanHeight_fit"):
    joined[col] = joined[col].multiply(INCH_TO_MM).round(2)

# Round digitization columns (seatWidth, seatDepth, panHeight) to 2 decimals
for col in ("seatWidth", "seatDepth", "panHeight"):
    if col in joined.columns:
        joined[col] = joined[col].round(2)

# Offsets: fitted - actual (in mm); blank if either value is missing
joined["seatWidth_offset"] = (joined["seatWidth_fit"] - joined["seatWidth"]).round(2)
joined["seatDepth_offset"] = (joined["seatDepth_fit"] - joined["seatDepth"]).round(2)
joined["panHeight_offset"] = (joined["seatPanHeight_fit"] - joined["panHeight"]).round(2)

# Column order: fit columns first, then digitization columns, then offsets at end
fit_cols = ["seatWidth_fit", "seatDepth_fit", "seatPanHeight_fit"]
digit_cols = ["seatWidth", "seatDepth", "panHeight"]
offset_cols = ["seatWidth_offset", "seatDepth_offset", "panHeight_offset"]
rest = [c for c in joined.columns if c not in fit_cols + digit_cols + offset_cols]
joined = joined[rest + fit_cols + digit_cols + offset_cols]

joined.to_csv("volunteer_with_digitizations.csv", index=False)