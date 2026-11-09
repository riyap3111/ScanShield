"""Step 8b: tune XGBoost and append it to the step-08 comparison tables.

Step 08 trains every family in one pass. This script exists so XGBoost can be added to an
existing run without repeating the Random Forest search, which is the expensive part. It uses
the same protocol as step 08 - randomized search, 5-fold stratified CV on the training split,
threshold chosen on validation, test split scored once afterwards - but with a smaller search
budget, which is recorded in the n_candidates column so the comparison stays honest.

The search runs in-process on purpose: libomp is loaded into this process, and joblib worker
subprocesses would each have to import xgboost themselves, which fails without that preload.

Usage:  python src/08b_add_xgboost.py [--n-iter 4]
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from config import (PROC_DIR, SPLIT_FILE, MET_DIR, TAB_DIR, TARGET,  # noqa: E402
                    RANDOM_STATE, TARGET_FPR)
from utils import save_table, save_json, select_model  # noqa: E402

MODEL_DIR = PROC_DIR / "models"


def load_step08():
    spec = importlib.util.spec_from_file_location("step08", SRC / "08_train_models.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["step08"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=4,
                    help="search budget; step 08 uses 15 for the other families")
    args = ap.parse_args()

    s8 = load_step08()
    if s8.XGB is None:
        sys.exit(f"XGBoost unavailable: {s8.XGB_NOTE}")

    df = pd.read_parquet(PROC_DIR / "model_matrix.parquet")
    mon = pd.read_parquet(PROC_DIR / "monday_matrix.parquet")
    splits = pd.read_parquet(SPLIT_FILE)
    fs = json.load(open(MET_DIR / "feature_sets_v2.json"))
    cols = fs["full"]
    masks = {p: (splits.strat_split == p).values for p in ["train", "val", "test"]}
    X = {p: df.loc[m, cols] for p, m in masks.items()}
    y = {p: df.loc[m, TARGET].values for p, m in masks.items()}

    est, space, rationale = s8.model_space()["XGBoost"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print(f"tuning XGBoost, {args.n_iter} candidates x 5 folds (in-process)")
    t0 = time.perf_counter()
    search = RandomizedSearchCV(est, space, n_iter=args.n_iter, cv=cv,
                                scoring="average_precision", random_state=RANDOM_STATE,
                                n_jobs=1, refit=True)
    search.fit(X["train"], y["train"])
    search_s = time.perf_counter() - t0
    best = search.best_estimator_
    params = {k: (v if isinstance(v, (str, int, float, type(None))) else str(v))
              for k, v in search.best_params_.items()}
    print(f"  best CV PR-AUC {search.best_score_:.6f} in {search_s:.0f}s")

    scoring = {"pr_auc": "average_precision", "roc_auc": "roc_auc",
               "recall": "recall", "precision": "precision", "f1": "f1"}
    cvres = cross_validate(best, X["train"], y["train"], cv=cv, scoring=scoring, n_jobs=1)
    cv_row = {"model": "XGBoost"}
    for k in scoring:
        cv_row[f"{k}_mean"], cv_row[f"{k}_std"] = cvres[f"test_{k}"].mean(), cvres[f"test_{k}"].std()

    t0 = time.perf_counter(); best.fit(X["train"], y["train"]); fit_s = time.perf_counter() - t0
    sval = best.predict_proba(X["val"])[:, 1]
    thr, _ = s8.pick_threshold(y["val"], sval)
    val = s8.metrics_at(y["val"], sval, thr)
    t0 = time.perf_counter(); stest = best.predict_proba(X["test"])[:, 1]
    infer_s = time.perf_counter() - t0
    test = s8.metrics_at(y["test"], stest, thr)
    mon_fpr = float((best.predict_proba(mon[cols])[:, 1] >= thr).mean())

    path = MODEL_DIR / "xgboost.joblib"
    joblib.dump(best, path)
    print(f"  VAL  R={val['recall']:.5f} FPR={val['fpr']:.5f} F1={val['f1']:.5f}")
    print(f"  TEST R={test['recall']:.5f} FPR={test['fpr']:.5f} F1={test['f1']:.5f} "
          f"PR-AUC={test['pr_auc']:.6f}  MondayFPR={mon_fpr:.6f}")

    def append(name, row, key="model"):
        tab = pd.read_csv(TAB_DIR / f"{name}.csv")
        tab = tab[tab[key] != "XGBoost"] if key in tab.columns else tab
        out = pd.concat([tab, pd.DataFrame([row])], ignore_index=True)
        save_table(out, name)
        return out

    comp = pd.read_csv(TAB_DIR / "model_comparison.csv")
    comp = comp[comp.model != "XGBoost"]
    rows = []
    for part, m in [("validation", val), ("test (locked)", test)]:
        r = {"model": "XGBoost", "split": part, "feature_set": "full", "n_features": len(cols), **m}
        r["meets_recall_target"] = m["recall"] >= 0.90
        r["meets_fpr_target"] = m["fpr"] <= TARGET_FPR
        rows.append(r)
    rows[-1]["monday_fpr"] = mon_fpr
    save_table(pd.concat([comp, pd.DataFrame(rows)], ignore_index=True), "model_comparison")

    append("model_tuning", {
        "model": "XGBoost", "n_candidates": args.n_iter, "cv_folds": 5,
        "search": "RandomizedSearchCV on train split, scored by average precision "
                  "(reduced budget, run in-process)",
        "best_params": json.dumps(params), "best_cv_pr_auc": search.best_score_,
        "search_seconds": search_s, "rationale": rationale})
    append("cv_variance", cv_row)
    append("threshold_choice", {
        "model": "XGBoost", "threshold": thr,
        "rule": f"max F1 subject to validation FPR <= {TARGET_FPR:.0%}",
        "val_fpr": val["fpr"], "val_recall": val["recall"], "val_f1": val["f1"],
        "test_fpr": test["fpr"], "test_recall": test["recall"]})
    append("compute_cost", {
        "model": "XGBoost", "fit_seconds_149k_rows": fit_s,
        "score_seconds_32k_rows": infer_s,
        "flows_scored_per_second": len(X["test"]) / infer_s,
        "model_size_mb": path.stat().st_size / 1e6, "search_seconds": search_s,
        "monday_fpr": mon_fpr, "monday_rows": len(mon)})

    save_json({"n_iter": args.n_iter, "best_params": params,
               "best_cv_pr_auc": search.best_score_, "threshold": thr,
               "validation": val, "test": test, "monday_fpr": mon_fpr}, "xgboost")

    print(f"\nselected model is now: {select_model()}")


if __name__ == "__main__":
    main()
