"""Step 4: Feature engineering, feature selection (fit on TRAIN only) and train/val/test splits.

Splits (all made AFTER de-duplication in step 2):
  * stratified : random 70/15/15, stratified on is_scan, random_state=42  (primary)
  * time       : first 70% / next 15% / last 15% of rows in original file order (time proxy,
                 this CSV has no timestamp column)
The TEST parts are saved but NOT used in this deliverable.

Outputs:
  data/processed/split_indices.parquet
  metrics/feature_sets.json
  tables/feature_selection_log.csv, split_summary.csv
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from config import (PORTSCAN_CLEAN, SPLIT_FILE, LABEL_COL, TARGET, PORT_COL, RANDOM_STATE,
                    CONSTANT_COLS, DUPLICATE_COLS)
from utils import save_table, save_json

NON_FEATURES = {LABEL_COL, TARGET, "raw_row"}
CORR_THRESHOLD = 0.95


def add_engineered(df):
    df["is_well_known_port"] = (df[PORT_COL] < 1024).astype(np.int8)
    return df


def make_splits(df):
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=df[TARGET], random_state=RANDOM_STATE)
    va, te = train_test_split(tmp, test_size=0.50, stratify=df[TARGET].iloc[tmp], random_state=RANDOM_STATE)
    strat = np.empty(len(df), dtype=object)
    strat[tr], strat[va], strat[te] = "train", "val", "test"

    order = np.argsort(df["raw_row"].values, kind="stable")
    n = len(df); a, b = int(n * .70), int(n * .85)
    time = np.empty(n, dtype=object)
    time[order[:a]], time[order[a:b]], time[order[b:]] = "train", "val", "test"
    return pd.DataFrame({"raw_row": df["raw_row"].values, "strat_split": strat, "time_split": time})


def select_features(train, candidates):
    """Greedy de-correlation fit on TRAIN only: rank by univariate AUC, keep a feature unless it
    has |r| > 0.95 with an already-kept, stronger feature."""
    y = train[TARGET].values
    auc = {}
    for f in candidates:
        x = train[f].values.astype(float)
        a = roc_auc_score(y, x) if x.std() > 0 else 0.5
        auc[f] = max(a, 1 - a)
    corr = train[candidates].corr().abs()
    ranked = sorted(candidates, key=lambda f: -auc[f])
    kept, log = [], []
    for f in ranked:
        clash = [k for k in kept if corr.loc[f, k] > CORR_THRESHOLD]
        if clash:
            k = max(clash, key=lambda c: corr.loc[f, c])
            log.append({"feature": f, "decision": "dropped", "train_auc": auc[f],
                        "reason": f"|r| = {corr.loc[f, k]:.3f} with kept feature '{k}' (AUC {auc[k]:.3f})"})
        else:
            kept.append(f)
            log.append({"feature": f, "decision": "kept", "train_auc": auc[f], "reason": "not redundant"})
    return kept, pd.DataFrame(log)


def main():
    df = add_engineered(pd.read_parquet(PORTSCAN_CLEAN))
    splits = make_splits(df)
    splits.to_parquet(SPLIT_FILE, index=False)

    # split summary
    rows = []
    for kind in ["strat_split", "time_split"]:
        for part in ["train", "val", "test"]:
            m = (splits[kind] == part).values
            y = df.loc[m, TARGET]
            rows.append({"split_type": kind.replace("_split", ""), "part": part, "rows": int(m.sum()),
                         "pct_of_data": m.mean() * 100, "PortScan": int(y.sum()),
                         "BENIGN": int((1 - y).sum()), "pct_PortScan": y.mean() * 100})
    summary = save_table(pd.DataFrame(rows), "split_summary")
    print(summary.to_string(index=False))

    # ---- Feature sets ---------------------------------------------------------------
    # all_no_port (PRIMARY): every original column except Destination Port, plus the two sentinel
    #   flags. Nothing is dropped, so no potentially useful signal is lost.
    # all_with_port: all_no_port + Destination Port + is_well_known_port (leakage check).
    # reduced_no_port: de-correlated subset (comparison only) to show whether pruning loses anything.
    port_feats = [PORT_COL, "is_well_known_port"]
    all_no_port = [c for c in df.columns if c not in NON_FEATURES and c not in port_feats]
    behavior = [c for c in all_no_port if c not in CONSTANT_COLS + DUPLICATE_COLS]
    train = df[(splits["strat_split"] == "train").values]
    kept, log = select_features(train, behavior)       # fit on stratified TRAIN only
    log = pd.concat([
        pd.DataFrame([{"feature": c, "decision": "reduced set: excluded (constant)", "train_auc": np.nan,
                       "reason": "Single value in raw file; kept in all-features sets"} for c in CONSTANT_COLS]),
        pd.DataFrame([{"feature": c, "decision": "reduced set: excluded (exact copy)", "train_auc": np.nan,
                       "reason": "Identical to another column; kept in all-features sets"} for c in DUPLICATE_COLS]),
        log,
        pd.DataFrame([
            {"feature": PORT_COL, "decision": "with_port set only", "train_auc": np.nan,
             "reason": "Leakage check: scanner's fixed port list may let a model memorise the lab setup"},
            {"feature": "is_well_known_port", "decision": "with_port set only (engineered)", "train_auc": np.nan,
             "reason": "Engineered: 1 if destination port < 1024"},
        ])])
    save_table(log, "feature_selection_log")

    sets = {"all_no_port": all_no_port, "all_with_port": all_no_port + port_feats,
            "reduced_no_port": kept,
            "n_original_columns_used": len([c for c in all_no_port if not c.startswith("init_win_")]) + 1,
            "n_candidates_for_reduction": len(behavior),
            "corr_threshold": CORR_THRESHOLD, "fit_on": "stratified train split only"}
    save_json(sets, "feature_sets")
    print(f"all_no_port={len(all_no_port)}, all_with_port={len(all_no_port)+2}, reduced_no_port={len(kept)} "
          f"(from {len(behavior)} distinct candidates)")


if __name__ == "__main__":
    main()
