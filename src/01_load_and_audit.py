"""Step 1: Load raw data and audit quality. Makes NO changes to the data.

Outputs:
  tables/data_dictionary.csv, data_quality_audit.csv, duplicate_summary.csv,
  outlier_summary.csv, class_distribution.csv, monday_audit.csv
  metrics/audit_facts.json
"""
import numpy as np
import pandas as pd

from config import (PORTSCAN_RAW, MONDAY_RAW, LABEL_COL, EXPECTED, KEY_FEATURES,
                    CONSTANT_COLS, DUPLICATE_COLS)
from utils import load_raw, save_table, save_json, feature_group, meaning


def audit_columns(df):
    num = df.drop(columns=LABEL_COL)
    rows = []
    for c in num.columns:
        s = num[c]
        rows.append({
            "column": c,
            "n_missing": int(s.isna().sum()),
            "n_inf": int(np.isinf(s).sum()),
            "n_negative": int((s < 0).sum()),
            "n_unique": int(s.nunique()),
            "is_constant": bool(s.nunique(dropna=False) <= 1),
        })
    return pd.DataFrame(rows)


def duplicate_columns(df):
    num = df.drop(columns=LABEL_COL)
    cols, dups = list(num.columns), []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if num[a].equals(num[b]) and num[a].nunique() > 1:
                dups.append((a, b))
    return dups


def main():
    df = load_raw(PORTSCAN_RAW)
    print(f"Loaded PortScan file: {df.shape}")

    # ---- Verify hand-checked facts --------------------------------------
    vc = df[LABEL_COL].value_counts()
    checks = {
        "rows": len(df), "cols": df.shape[1],
        "PortScan": int(vc.get("PortScan", 0)), "BENIGN": int(vc.get("BENIGN", 0)),
        "dup_rows": int(df.duplicated().sum()),
    }
    for k, v in EXPECTED.items():
        assert checks[k] == v, f"Fact check failed for {k}: expected {v}, got {checks[k]}"
    print("All hand-checked facts match.")

    # ---- Data dictionary -------------------------------------------------
    num = df.drop(columns=LABEL_COL)
    dd = []
    for c in df.columns:
        s = df[c]
        role = "feature"
        if c == LABEL_COL: role = "target"
        elif c in CONSTANT_COLS: role = "feature (all-features sets); excluded from reduced set (constant)"
        elif c in DUPLICATE_COLS: role = "feature (all-features sets); excluded from reduced set (exact copy)"
        elif c == "Destination Port": role = "feature (all_with_port set only; leakage check)"
        is_num = c != LABEL_COL
        finite = s.replace([np.inf, -np.inf], np.nan) if is_num else s
        dd.append({
            "column": c,
            "dtype": str(s.dtype),
            "type": "categorical (binary)" if c == LABEL_COL else
                    ("numeric (discrete)" if "int" in str(s.dtype) else "numeric (continuous)"),
            "group": feature_group(c),
            "meaning": meaning(c),
            "non_null": int(s.notna().sum()),
            "n_unique": int(s.nunique()),
            "min": float(finite.min()) if is_num else None,
            "median": float(finite.median()) if is_num else None,
            "max": float(finite.max()) if is_num else None,
            "role": role,
        })
    save_table(pd.DataFrame(dd), "data_dictionary")

    # ---- Column-level quality -------------------------------------------
    colq = audit_columns(df)
    dcols = duplicate_columns(df)
    colq["duplicate_of"] = colq["column"].map({b: a for a, b in dcols})
    save_table(colq, "data_quality_by_column")

    # ---- Duplicate rows & conflicting labels -----------------------------
    feats = list(num.columns)
    dup_all = df.duplicated()
    dup_by_label = df[dup_all][LABEL_COL].value_counts()
    grp = df.groupby(feats, dropna=False)[LABEL_COL].nunique()
    conflict_groups = int((grp > 1).sum())
    conflict_mask = df.set_index(feats).index.isin(grp[grp > 1].index)
    conflict_rows = int(conflict_mask.sum())
    dup_summary = pd.DataFrame([
        {"metric": "exact duplicate rows (all columns)", "value": int(dup_all.sum()),
         "pct_of_rows": dup_all.mean() * 100},
        {"metric": "  of which PortScan", "value": int(dup_by_label.get("PortScan", 0)),
         "pct_of_rows": dup_by_label.get("PortScan", 0) / vc["PortScan"] * 100},
        {"metric": "  of which BENIGN", "value": int(dup_by_label.get("BENIGN", 0)),
         "pct_of_rows": dup_by_label.get("BENIGN", 0) / vc["BENIGN"] * 100},
        {"metric": "feature-vectors with conflicting labels (groups)", "value": conflict_groups,
         "pct_of_rows": np.nan},
        {"metric": "rows inside conflicting groups", "value": conflict_rows,
         "pct_of_rows": conflict_rows / len(df) * 100},
    ])
    save_table(dup_summary, "duplicate_summary")

    # ---- Quality overview table -----------------------------------------
    neg = colq[colq.n_negative > 0][["column", "n_negative"]]
    quality = pd.DataFrame([
        {"issue": "Missing values (NaN)", "evidence": f"{int(colq.n_missing.sum())} cells, all in "
            + ", ".join(colq[colq.n_missing > 0].column)},
        {"issue": "Infinite values", "evidence": "; ".join(f"{r.column}: {r.n_inf}" for r in colq[colq.n_inf > 0].itertuples())},
        {"issue": "Negative values", "evidence": "; ".join(f"{r.column}: {r.n_negative}" for r in neg.itertuples())},
        {"issue": "Constant columns", "evidence": f"{int(colq.is_constant.sum())}: " + ", ".join(colq[colq.is_constant].column)},
        {"issue": "Duplicate columns", "evidence": "; ".join(f"{b} == {a}" for a, b in dcols)},
        {"issue": "Exact duplicate rows", "evidence": f"{int(dup_all.sum())} ({dup_all.mean()*100:.1f}%)"},
        {"issue": "Conflicting labels", "evidence": f"{conflict_groups} feature-vectors appear with both labels ({conflict_rows} rows)"},
    ])
    save_table(quality, "data_quality_audit")

    # ---- Outliers on key features (IQR fences, per class) ----------------
    out = []
    for c in KEY_FEATURES:
        for lab, g in df.groupby(LABEL_COL):
            s = g[c].replace([np.inf, -np.inf], np.nan).dropna()
            q1, q3 = s.quantile([.25, .75])
            iqr = q3 - q1
            pct = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean() * 100
            out.append({"feature": c, "label": lab, "pct_outside_IQR_fences": pct,
                        "p1": s.quantile(.01), "p50": s.median(), "p99": s.quantile(.99), "max": s.max()})
    save_table(pd.DataFrame(out), "outlier_summary")

    # ---- Class distribution raw vs dedup ---------------------------------
    dedup = df.drop_duplicates()
    cd = pd.DataFrame({
        "raw_count": vc, "raw_pct": vc / vc.sum() * 100,
        "dedup_count": dedup[LABEL_COL].value_counts(),
    })
    cd["dedup_pct"] = cd.dedup_count / cd.dedup_count.sum() * 100
    cd["removed_pct"] = (1 - cd.dedup_count / cd.raw_count) * 100
    cd.index.name = "label"
    save_table(cd.reset_index(), "class_distribution")

    # ---- Row-order structure (time proxy) --------------------------------
    s = (df[LABEL_COL] == "PortScan").astype(int).values
    idx = np.where(s)[0]

    # ---- Monday benign file ----------------------------------------------
    mon = load_raw(MONDAY_RAW)
    mnum = mon.drop(columns=LABEL_COL)
    mon_audit = pd.DataFrame([{
        "rows": len(mon), "cols": mon.shape[1],
        "labels": mon[LABEL_COL].value_counts().to_dict(),
        "dup_rows": int(mon.duplicated().sum()),
        "nan_cells": int(mnum.isna().sum().sum()),
        "inf_cells": int(np.isinf(mnum).sum().sum()),
        "neg_flow_duration": int((mon["Flow Duration"] < 0).sum()),
        "same_columns_as_portscan": list(mon.columns) == list(df.columns),
    }])
    save_table(mon_audit, "monday_audit")

    facts = {
        **checks,
        "label_pct": (vc / vc.sum() * 100).round(2).to_dict(),
        "nan_cells": int(colq.n_missing.sum()), "inf_cells": int(colq.n_inf.sum()),
        "constant_cols": colq[colq.is_constant].column.tolist(),
        "duplicate_cols": dcols,
        "conflict_groups": conflict_groups, "conflict_rows": conflict_rows,
        "scan_row_range": [int(idx.min()), int(idx.max())],
        "label_switches_in_row_order": int((np.diff(s) != 0).sum()),
        "distinct_ports": df.groupby(LABEL_COL)["Destination Port"].nunique().to_dict(),
        "monday": mon_audit.iloc[0].to_dict(),
    }
    save_json(facts, "audit_facts")
    print(quality.to_string(index=False))
    print(dup_summary.to_string(index=False))
    print(cd)


if __name__ == "__main__":
    main()
