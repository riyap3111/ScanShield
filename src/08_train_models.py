"""Step 8 (Deliverable 3): Model development, tuning and final scoring.

Four model families, all trained on the SAME stratified training split and the SAME primary
feature set ('full' - every original column plus the engineered features):

  1. Logistic regression          linear, the Deliverable 2 reference baseline
  2. Random Forest                bagged trees, non-linear, handles interactions
  3. Histogram Gradient Boosting  boosted trees, scikit-learn implementation
  4. XGBoost                      boosted trees, the library named in the proposal

XGBoost needs an OpenMP runtime that Apple does not ship. src/install_libomp.py fetches it from
conda-forge into ~/.local/lib and src/xgb_support.py loads it before importing xgboost; if it is
missing, XGBoost is skipped and the other three still run.

Protocol
  tuning      RandomizedSearchCV, 5-fold StratifiedKFold, on the TRAINING split only,
              scored by average precision (PR-AUC), which is the right ranking metric under
              class imbalance.
  threshold   chosen on the VALIDATION split: the threshold that maximises F1 subject to
              false-positive rate <= 5% (the proposal's operational ceiling). Never on test.
  test        the locked test split is scored ONCE, at the end, with the frozen threshold.
  imbalance   class weights (balanced), not resampling: the imbalance is mild (42.5% scan).
  variance    5-fold CV mean +/- std, plus the final model refit under 3 random seeds.

Outputs:
  data/processed/models/*.joblib
  metrics/models_v2.json, metrics/cv_results.json, metrics/cost.json
  tables/model_comparison.csv, model_tuning.csv, cv_variance.csv, threshold_choice.csv,
         feature_set_ablation.csv, seed_stability.csv
"""
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix)

from config import PROC_DIR, SPLIT_FILE, MET_DIR, TARGET, RANDOM_STATE, TARGET_RECALL, TARGET_FPR
from utils import save_table, save_json, signed_log1p
from xgb_support import load_xgboost

XGB, XGB_NOTE = load_xgboost()

MODEL_MATRIX = PROC_DIR / "model_matrix.parquet"
MONDAY_MATRIX = PROC_DIR / "monday_matrix.parquet"
MODEL_DIR = PROC_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# The two classes are almost linearly separable, so the weakly regularised end of the logistic
# regression search drives the weights large enough to overflow in float64. lbfgs recovers on
# its own and the fitted model is unaffected; the warnings are just noise in the log.
warnings.filterwarnings("ignore", message=".*overflow encountered.*")
warnings.filterwarnings("ignore", message=".*divide by zero encountered.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered.*")
np.seterr(over="ignore", divide="ignore", invalid="ignore")

N_ITER = 15
CV_FOLDS = 5
POS_WEIGHT = 1.353        # negatives/positives in the training split, for scale_pos_weight
N_JOBS = 4          # fixed pool size; -1 oversubscribes and stalls on this machine
SEEDS = [42, 7, 2024]
PRIMARY_SET = "full"


def lr_pipeline(random_state=RANDOM_STATE):
    return Pipeline([
        ("log", FunctionTransformer(signed_log1p)),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=random_state)),
    ])


def model_space(random_state=RANDOM_STATE):
    """(estimator, search space, why this family) per model."""
    space = {
        "Logistic regression": (
            lr_pipeline(random_state),
            {"lr__C": loguniform(1e-3, 1e2)},
            "Linear reference baseline: fast, fully interpretable coefficients, carried over from "
            "Deliverable 2 so the new models have a like-for-like comparator.",
        ),
        "Random Forest": (
            # n_jobs=1 here on purpose: the search below is what runs in parallel, and nesting
            # joblib pools inside each other deadlocks on macOS.
            RandomForestClassifier(n_jobs=1, random_state=random_state),
            {"n_estimators": randint(150, 400),
             "max_depth": [None, 8, 12, 16, 24],
             "min_samples_leaf": randint(1, 20),
             "max_features": ["sqrt", "log2", 0.3],
             "class_weight": ["balanced", "balanced_subsample"]},
            "Bagged trees: captures threshold interactions a linear model cannot, resists "
            "overfitting through averaging, and gives impurity and permutation importances.",
        ),
        "Gradient Boosting": (
            HistGradientBoostingClassifier(random_state=random_state, early_stopping=False),
            {"learning_rate": loguniform(0.02, 0.3),
             "max_iter": randint(100, 400),
             "max_leaf_nodes": randint(15, 63),
             "min_samples_leaf": randint(10, 60),
             "l2_regularization": uniform(0, 2.0),
             "class_weight": ["balanced"]},
            "Boosted trees: fits residual errors in sequence; scikit-learn's histogram "
            "implementation.",
        ),
    }
    if XGB is not None:
        # scale_pos_weight is XGBoost's equivalent of balanced class weights.
        space["XGBoost"] = (
            XGB.XGBClassifier(tree_method="hist", n_jobs=1, random_state=random_state,
                              eval_metric="aucpr", verbosity=0),
            {"n_estimators": randint(150, 450),
             "max_depth": randint(3, 10),
             "learning_rate": loguniform(0.02, 0.3),
             "subsample": uniform(0.6, 0.4),
             "colsample_bytree": uniform(0.6, 0.4),
             "min_child_weight": randint(1, 20),
             "reg_lambda": uniform(0, 3.0),
             "scale_pos_weight": [1.0, POS_WEIGHT]},
            "The gradient-boosting library named in the proposal, with the histogram tree "
            "method.",
        )
    return space


def metrics_at(y, score, thr):
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
        "roc_auc": roc_auc_score(y, score) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, score) if len(np.unique(y)) > 1 else np.nan,
        "accuracy": (tp + tn) / len(y),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def pick_threshold(y, score, max_fpr=TARGET_FPR):
    """Highest-F1 threshold whose false-positive rate stays within the operational ceiling.

    Chosen on validation only. Falls back to the lowest-FPR threshold if nothing meets the
    ceiling (cannot happen at current performance, but keeps the function total)."""
    grid = np.unique(np.quantile(score, np.linspace(0.001, 0.999, 400)))
    rows = []
    for t in grid:
        pred = (score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        rows.append((t, f1_score(y, pred, zero_division=0), fpr,
                     recall_score(y, pred, zero_division=0)))
    tab = pd.DataFrame(rows, columns=["threshold", "f1", "fpr", "recall"])
    ok = tab[tab.fpr <= max_fpr]
    best = (ok.sort_values("f1", ascending=False).iloc[0] if len(ok)
            else tab.sort_values("fpr").iloc[0])
    return float(best.threshold), tab


def scores_of(model, X):
    return model.predict_proba(X)[:, 1]


def main():
    df = pd.read_parquet(MODEL_MATRIX)
    mon = pd.read_parquet(MONDAY_MATRIX)
    splits = pd.read_parquet(SPLIT_FILE)
    assert (splits.raw_row.values == df.raw_row.values).all()
    fs = json.load(open(MET_DIR / "feature_sets_v2.json"))
    cols = fs[PRIMARY_SET]

    masks = {p: (splits.strat_split == p).values for p in ["train", "val", "test"]}
    X = {p: df.loc[m, cols] for p, m in masks.items()}
    y = {p: df.loc[m, TARGET].values for p, m in masks.items()}
    Xmon, ymon = mon[cols], mon[TARGET].values
    assert ymon.sum() == 0, "Monday must be all benign"
    print(f"train {X['train'].shape}  val {X['val'].shape}  test {X['test'].shape}  "
          f"monday {Xmon.shape}  |  {len(cols)} features")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    tuning_rows, cv_rows, comparison, thresholds, cost_rows = [], [], [], [], []
    fitted, cv_detail = {}, {}

    for name, (est, space, rationale) in model_space().items():
        print(f"\n=== {name} ===")
        t0 = time.perf_counter()
        search = RandomizedSearchCV(est, space, n_iter=N_ITER, cv=cv, scoring="average_precision",
                                    random_state=RANDOM_STATE, n_jobs=N_JOBS, refit=True,
                                    return_train_score=True)
        search.fit(X["train"], y["train"])
        search_s = time.perf_counter() - t0
        best = search.best_estimator_
        params = {k: (v if isinstance(v, (str, int, float, type(None))) else str(v))
                  for k, v in search.best_params_.items()}
        print(f"  best CV PR-AUC {search.best_score_:.6f} in {search_s:.0f}s  params {params}")

        tuning_rows.append({
            "model": name, "n_candidates": N_ITER, "cv_folds": CV_FOLDS,
            "search": "RandomizedSearchCV on train split, scored by average precision",
            "best_params": json.dumps(params), "best_cv_pr_auc": search.best_score_,
            "search_seconds": search_s, "rationale": rationale,
        })

        # 5-fold CV with the tuned configuration -> mean +/- std for several metrics
        scoring = {"pr_auc": "average_precision", "roc_auc": "roc_auc",
                   "recall": "recall", "precision": "precision", "f1": "f1"}
        cvres = cross_validate(best, X["train"], y["train"], cv=cv, scoring=scoring, n_jobs=N_JOBS)
        row = {"model": name}
        for k in scoring:
            v = cvres[f"test_{k}"]
            row[f"{k}_mean"], row[f"{k}_std"] = v.mean(), v.std()
        cv_rows.append(row)
        cv_detail[name] = {k: cvres[f"test_{k}"].tolist() for k in scoring}
        print("  CV " + "  ".join(f"{k}={row[k+'_mean']:.5f}+/-{row[k+'_std']:.5f}" for k in scoring))

        # refit timing on the full training split, then the frozen operating point
        if hasattr(best, "n_jobs"):
            best.set_params(n_jobs=N_JOBS)
        t0 = time.perf_counter(); best.fit(X["train"], y["train"]); fit_s = time.perf_counter() - t0
        sval = scores_of(best, X["val"])
        thr, curve = pick_threshold(y["val"], sval)
        val = metrics_at(y["val"], sval, thr)

        t0 = time.perf_counter(); stest = scores_of(best, X["test"]); infer_s = time.perf_counter() - t0
        test = metrics_at(y["test"], stest, thr)
        smon = scores_of(best, Xmon)
        mon_fpr = float((smon >= thr).mean())

        size_mb = None
        p = MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(best, p); size_mb = p.stat().st_size / 1e6
        fitted[name] = (best, thr, sval, stest)

        for part, m in [("validation", val), ("test (locked)", test)]:
            comparison.append({"model": name, "split": part, "feature_set": PRIMARY_SET,
                               "n_features": len(cols), **m})
        comparison[-1]["monday_fpr"] = mon_fpr
        thresholds.append({"model": name, "threshold": thr,
                           "rule": f"max F1 subject to validation FPR <= {TARGET_FPR:.0%}",
                           "val_fpr": val["fpr"], "val_recall": val["recall"],
                           "val_f1": val["f1"], "test_fpr": test["fpr"], "test_recall": test["recall"]})
        cost_rows.append({
            "model": name, "fit_seconds_149k_rows": fit_s,
            "score_seconds_32k_rows": infer_s,
            "flows_scored_per_second": len(X["test"]) / infer_s,
            "model_size_mb": size_mb, "search_seconds": search_s,
            "monday_fpr": mon_fpr, "monday_rows": len(Xmon),
        })
        print(f"  VAL  P={val['precision']:.5f} R={val['recall']:.5f} F1={val['f1']:.5f} "
              f"FPR={val['fpr']:.5f} PR-AUC={val['pr_auc']:.6f}")
        print(f"  TEST P={test['precision']:.5f} R={test['recall']:.5f} F1={test['f1']:.5f} "
              f"FPR={test['fpr']:.5f} PR-AUC={test['pr_auc']:.6f}  MondayFPR={mon_fpr:.6f}")
        if name == "Gradient Boosting":
            save_table(curve, "threshold_curve_gb")

    comp = pd.DataFrame(comparison)
    for c in ["meets_recall_target", "meets_fpr_target"]:
        comp[c] = comp["recall"] >= TARGET_RECALL if c.startswith("meets_recall") else comp["fpr"] <= TARGET_FPR
    save_table(comp, "model_comparison")
    save_table(pd.DataFrame(tuning_rows), "model_tuning")
    save_table(pd.DataFrame(cv_rows), "cv_variance")
    save_table(pd.DataFrame(thresholds), "threshold_choice")
    save_table(pd.DataFrame(cost_rows), "compute_cost")

    # The selected family is decided on VALIDATION, never on test, and the two studies below
    # use that same family so they describe the model the report actually recommends.
    val_rows = pd.DataFrame([r for r in comparison if r["split"] == "validation"])
    eligible = val_rows[val_rows.fpr <= TARGET_FPR]
    eligible = eligible if len(eligible) else val_rows
    selected = eligible.sort_values(["f1", "pr_auc"], ascending=False).iloc[0]["model"]
    sel_est = fitted[("strat", selected)][0]
    print(f"\nselected on validation: {selected}")

    # ---- feature-set ablation: does engineering help, and how much is tool fingerprint? ----
    print(f"=== feature-set ablation ({selected}, tuned config) ===")
    abl = []
    for set_name in ["full", "original_only", "no_fingerprint"]:
        c = fs[set_name]
        m = clone(sel_est)
        m.fit(df.loc[masks["train"], c], y["train"])
        sv = m.predict_proba(df.loc[masks["val"], c])[:, 1]
        t, _ = pick_threshold(y["val"], sv)
        v = metrics_at(y["val"], sv, t)
        mf = float((m.predict_proba(mon[c])[:, 1] >= t).mean())
        abl.append({"feature_set": set_name, "n_features": len(c), **v, "monday_fpr": mf})
        print(f"  {set_name:15s} n={len(c):3d} R={v['recall']:.5f} FPR={v['fpr']:.5f} "
              f"PR-AUC={v['pr_auc']:.6f} MondayFPR={mf:.6f}")
    save_table(pd.DataFrame(abl), "feature_set_ablation")

    # ---- seed stability of the selected family ----------------------------------------------
    print(f"\n=== seed stability ({selected}, validation) ===")
    seed_rows = []
    for s in SEEDS:
        m = clone(sel_est)
        m.set_params(**{k: s for k in m.get_params() if k.endswith("random_state")})
        m.fit(X["train"], y["train"])
        sv = m.predict_proba(X["val"])[:, 1]
        t, _ = pick_threshold(y["val"], sv)
        v = metrics_at(y["val"], sv, t)
        seed_rows.append({"seed": s, **{k: v[k] for k in
                                        ["precision", "recall", "f1", "fpr", "pr_auc", "roc_auc"]}})
        print(f"  seed {s}: R={v['recall']:.5f} FPR={v['fpr']:.5f} F1={v['f1']:.5f}")
    sd = pd.DataFrame(seed_rows)
    sd.loc[len(sd)] = {"seed": "mean", **sd.drop(columns="seed").mean().to_dict()}
    sd.loc[len(sd)] = {"seed": "std", **pd.DataFrame(seed_rows).drop(columns="seed").std().to_dict()}
    save_table(sd, "seed_stability")

    save_json({"primary_feature_set": PRIMARY_SET, "n_features": len(cols),
               "xgboost": XGB_NOTE if XGB is None else f"xgboost {XGB.__version__}: {XGB_NOTE}",
               "tuning": "RandomizedSearchCV, 5-fold StratifiedKFold, average precision, train split only",
               "threshold_rule": f"max F1 subject to validation FPR <= {TARGET_FPR}",
               "test_scored_once": True, "seeds": SEEDS,
               "selected_model": selected,
               "selection_rule": "highest validation F1 within the FPR ceiling, PR-AUC as tie-break",
               "results": comparison, "cost": cost_rows}, "models_v2")
    save_json(cv_detail, "cv_results")
    print("\nSaved model_comparison, model_tuning, cv_variance, threshold_choice, "
          "compute_cost, feature_set_ablation, seed_stability.")


if __name__ == "__main__":
    main()
