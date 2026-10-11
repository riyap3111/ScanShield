"""Step 2: Clean the data. Every step is logged with row/column counts before -> after.

Outputs:
  data/processed/portscan_clean.parquet, monday_benign_clean.parquet
  tables/cleaning_log.csv, tables/cleaning_log_monday.csv
"""
import numpy as np
import pandas as pd

from config import (PORTSCAN_RAW, MONDAY_RAW, PORTSCAN_CLEAN, MONDAY_CLEAN, LABEL_COL,
                    TARGET, POS_LABEL, CONSTANT_COLS, DUPLICATE_COLS)
from utils import load_raw, save_table


def clean(df, log_name):
    log = []

    def record(step, rule, rationale, before, after, affected=""):
        log.append({"step": step, "rule": rule,
                    "rows_before": before[0], "rows_after": after[0],
                    "cols_before": before[1], "cols_after": after[1],
                    "rows_removed": before[0] - after[0],
                    "removed_by_label": affected, "rationale": rationale})

    def by_label(mask):
        vc = df.loc[mask, LABEL_COL].value_counts().to_dict()
        return ", ".join(f"{k}: {v}" for k, v in vc.items()) or "none"

    # 1. strip column names (load_raw already does it; logged for traceability)
    b = df.shape
    df.columns = df.columns.str.strip()
    record(1, "Strip whitespace from column names", "Raw headers have leading spaces, which break name lookups", b, df.shape)

    # 2. inf -> NaN, then drop NaN rows
    b = df.shape
    feats = [c for c in df.columns if c != LABEL_COL]
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    bad = df[feats].isna().any(axis=1)
    aff = by_label(bad)
    df = df[~bad].copy()
    record(2, "Replace +/-inf with NaN, drop rows with any NaN",
           "Inf/NaN come from zero-duration flows (division by 0 in Flow Bytes/s, Flow Packets/s); "
           "rate cannot be computed so the row is unusable", b, df.shape, aff)

    # 3. impossible negatives in duration / IAT
    b = df.shape
    bad = (df["Flow Duration"] < 0) | (df["Flow IAT Min"] < 0)
    aff = by_label(bad)
    df = df[~bad].copy()
    record(3, "Drop rows with Flow Duration < 0 or Flow IAT Min < 0",
           "Durations and inter-arrival times cannot be negative (clock/parsing errors in CICFlowMeter)", b, df.shape, aff)

    # 4. -1 sentinel flags in TCP window columns (kept, not dropped)
    b = df.shape
    df["init_win_fwd_missing"] = (df["Init_Win_bytes_forward"] == -1).astype(np.int8)
    df["init_win_bwd_missing"] = (df["Init_Win_bytes_backward"] == -1).astype(np.int8)
    record(4, "Add binary flags for Init_Win_bytes_* == -1 (values kept)",
           "-1 means 'window not observed', not an error; dropping would remove 21% (fwd) to 27% (bwd) of PortScan-file rows. "
           "The flag lets models use 'missing' as information", b, df.shape)

    # 5. constant + duplicate columns are FLAGGED, not dropped, so the all-features set keeps every
    #    original column (team decision: do not risk losing a useful feature). They are only
    #    excluded from the 'reduced' comparison set built in step 04.
    b = df.shape
    flagged = [c for c in CONSTANT_COLS + DUPLICATE_COLS if c in df.columns]
    record(5, f"Flag (keep) {len(flagged)} redundant columns: 10 constant + 8 exact copies",
           "Kept so the all-features models see every original column; excluded only from the reduced "
           "comparison feature set (e.g. Subflow Fwd Packets == Total Fwd Packets, Fwd PSH Flags == SYN Flag Count)",
           b, df.shape)

    # 6. exact duplicate rows
    b = df.shape
    dup = df.duplicated(keep="first")
    aff = by_label(dup)
    df = df[~dup].copy()
    record(6, "Remove exact duplicate rows (keep first) BEFORE splitting",
           "Copies could land in both train and test, inflating scores (leakage)", b, df.shape, aff)

    # 7. conflicting-label groups
    b = df.shape
    feats = [c for c in df.columns if c != LABEL_COL]
    n_lab = df.groupby(feats, dropna=False)[LABEL_COL].transform("nunique")
    bad = n_lab > 1
    aff = by_label(bad)
    df = df[~bad].copy()
    record(7, "Drop all rows whose identical feature-vector appears with both labels",
           "Same input with two labels is label noise; no model can get both right", b, df.shape, aff)

    # 8. encode target
    b = df.shape
    df[TARGET] = (df[LABEL_COL] == POS_LABEL).astype(np.int8)
    record(8, f"Encode target: {TARGET} = 1 if PortScan else 0",
           "Binary target for classification", b, df.shape)

    df = df.reset_index(drop=False).rename(columns={"index": "raw_row"})
    log_df = pd.DataFrame(log)
    save_table(log_df, log_name)
    return df, log_df


def main():
    ps, log = clean(load_raw(PORTSCAN_RAW), "cleaning_log")
    ps.to_parquet(PORTSCAN_CLEAN, index=False)
    print(log[["step", "rows_before", "rows_after", "cols_after", "removed_by_label"]].to_string(index=False))
    print(ps[LABEL_COL].value_counts())

    mon, mlog = clean(load_raw(MONDAY_RAW), "cleaning_log_monday")
    mon.to_parquet(MONDAY_CLEAN, index=False)
    print(mlog[["step", "rows_before", "rows_after"]].to_string(index=False))


if __name__ == "__main__":
    main()
