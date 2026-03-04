import re
import pandas as pd

# Columns to keep when loading (data cleaning)
COLUMNS = ["ID", "Gender", "Age", "Height (cm)", "BMI", "WC Type"]


def _normalize_id(id_str: str) -> str:
    """Normalize ID to match digitizations: remove '_', then strip after first letters+digits."""
    s = str(id_str).strip().replace("_", "")
    match = re.match(r"^([A-Za-z]+\d+).*", s)
    return match.group(1) if match else s


class XlsxReader:
    """Reads an Excel file and holds the result as a DataFrame member."""

    def __init__(self):
        self.df: pd.DataFrame | None = None

    def load(self, path: str, sheet_name=0, **kwargs) -> pd.DataFrame:
        """
        Load an Excel file and set it as the class member df.
        Keeps only ID, Gender, Age, Height (cm), and BMI columns.

        Args:
            path: Path to the .xlsx file.
            sheet_name: Sheet index or name (default: first sheet).
            **kwargs: Extra arguments passed to pd.read_excel (e.g. skiprows).

        Returns:
            The loaded DataFrame (also stored in self.df).
        """
        self.df = pd.read_excel(
            path, sheet_name=sheet_name, usecols=COLUMNS, **kwargs
        )
        # Split ID by "/" and put each ID on its own row (duplicate other cols)
        self.df["ID"] = self.df["ID"].astype(str).str.split("/")
        self.df = self.df.explode("ID", ignore_index=True)
        self.df["ID"] = self.df["ID"].str.strip()
        # Normalize ID (remove '_', strip after letters+digits) to match digitizations
        self.df["ID"] = self.df["ID"].apply(_normalize_id)
        return self.df
