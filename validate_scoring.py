from __future__ import annotations

import sys
from pathlib import Path

from scoring_engine import load_workbook_dataset

if len(sys.argv) < 2:
    raise SystemExit("Usage: python validate_scoring.py /path/to/workbook.xlsx")

path = Path(sys.argv[1])
dataset = load_workbook_dataset(path, path.name)
print("Records:", len(dataset.df))
print("Long scored rows:", len(dataset.long_df))
print("Compliance rows:", len(dataset.compliance_df))
print("Diagnostics:")
for key, value in dataset.diagnostics.items():
    if key in {"raw_columns_used", "unknown_scoring_values"}:
        continue
    print(f"  {key}: {value}")
print(dataset.df[["Record ID", "Agent Name", "Quality Auditor", "Call/Chat", "Overall Score (Without Fatal)", "Overall Score (With Fatal)", "Defect %", "Has Fatal"]].head(10))
