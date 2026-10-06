#!/usr/bin/env python3
"""Split the auto CSRC sheet into tight (window <= max_days) and all; drop conduct windows ending before the price panel."""
import sys
import pandas as pd

src, max_days, panel_start = sys.argv[1], int(sys.argv[2]), pd.Timestamp(sys.argv[3])
sh = pd.read_csv(src, dtype={"ticker": str})
span = (pd.to_datetime(sh["end_date"]) - pd.to_datetime(sh["onset_date"])).dt.days
sh = sh[pd.to_datetime(sh["end_date"]) >= panel_start]
tight = sh[span.loc[sh.index] <= max_days]
sh.to_csv(src.replace(".csv", "_all.csv"), index=False)
tight.to_csv(src.replace(".csv", "_tight.csv"), index=False)
print("all", len(sh), "rows", sh["case_id"].nunique(), "cases | tight", len(tight), "rows", tight["case_id"].nunique(), "cases")
print(tight.groupby("anomaly_class")["case_id"].nunique().to_dict())
