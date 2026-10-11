"""Step 5: Preliminary baselines. All metrics on the VALIDATION part only; TEST stays locked.

Models: majority class, rule-based detector (thresholds tuned on train), logistic regression
(no_port and with_port), depth-4 decision tree. Each is run on the stratified and the
time-ordered split, and also scored on Monday (all-benign, different day) for false-positive rate.

Outputs:
  metrics/baselines.json
  tables/baseline_results.csv, rule_definition.csv, error_analysis.csv, lr_coefficients.csv
  figures/fig6_confusion_matrix.png, fig7_roc_pr_curves.png, fig8_lr_coefficients.png,
          fig9_decision_tree.png
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, roc_curve,
                             precision_recall_curve)

from config import (PORTSCAN_CLEAN, MONDAY_CLEAN, SPLIT_FILE, MET_DIR, TARGET, PORT_COL,
                    RANDOM_STATE, TARGET_RECALL, TARGET_FPR, COLORS)
from utils import save_table, save_json, save_fig, style, Timer

style()


def signed_log1p(X):
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


def lr_pipeline():
    return Pipeline([
        ("log", FunctionTransformer(signed_log1p)),
        ("scale", StandardScaler()),          # fit on train only (inside the pipeline)
        ("lr", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE)),
    ])


# ---------------------------------------------------------------- rule-based detector
class RuleDetector:
    """Flag a flow as a scan if it is SHORT, has FEW forward packets and carries NO/LITTLE payload.
    Thresholds chosen on train by maximising (recall - FPR) over a small grid."""
    def fit(self, X, y):
        best = None
        for d in [100, 500, 1_000, 5_000, 10_000, 50_000]:
            for n in [1, 2, 3]:
                for p in [0, 2, 6]:
                    pred = self._apply(X, d, n, p)
                    rec = pred[y == 1].mean(); fpr = pred[y == 0].mean()
                    if best is None or rec - fpr > best[0]:
                        best = (rec - fpr, d, n, p)
        _, self.d, self.n, self.p = best
        return self

    @staticmethod
    def _apply(X, d, n, p):
        return ((X["Flow Duration"] <= d) & (X["Total Fwd Packets"] <= n)
                & (X["Fwd Packet Length Max"] <= p)).astype(int).values

    def predict(self, X): return self._apply(X, self.d, self.n, self.p)
    def score(self, X): return self.predict(X).astype(float)

    def describe(self):
        return (f"Flag as PortScan if Flow Duration <= {self.d:,} microseconds AND Total Fwd Packets "
                f"<= {self.n} AND Fwd Packet Length Max <= {self.p} bytes")


class Majority:
    def fit(self, X, y): self.c = int(round(y.mean())) ; return self
    def predict(self, X): return np.full(len(X), self.c)
    def score(self, X): return np.full(len(X), float(self.c))


class SkWrap:
    def __init__(self, m): self.m = m
    def fit(self, X, y): self.m.fit(X, y); return self
    def predict(self, X): return self.m.predict(X)
    def score(self, X): return self.m.predict_proba(X)[:, 1]


def metrics(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    one_class_score = len(np.unique(score)) < 2
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "fpr": fp / (fp + tn),
        "roc_auc": np.nan if one_class_score else roc_auc_score(y, score),
        "pr_auc": np.nan if one_class_score else average_precision_score(y, score),
        "accuracy": (tp + tn) / len(y),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def main():
    df = pd.read_parquet(PORTSCAN_CLEAN)
    df["is_well_known_port"] = (df[PORT_COL] < 1024).astype(np.int8)
    mon = pd.read_parquet(MONDAY_CLEAN)
    mon["is_well_known_port"] = (mon[PORT_COL] < 1024).astype(np.int8)
    assert mon[TARGET].sum() == 0, "Monday file should be all benign"
    splits = pd.read_parquet(SPLIT_FILE)
    assert (splits.raw_row.values == df.raw_row.values).all()
    fs = json.load(open(MET_DIR / "feature_sets.json"))

    models = {
        "Majority class": (lambda: Majority(), "all_no_port"),
        "Rule-based (duration/packets/payload)": (lambda: RuleDetector(), "all_no_port"),
        "Logistic regression (all features, no port)": (lambda: SkWrap(lr_pipeline()), "all_no_port"),
        "Logistic regression (all features + port)": (lambda: SkWrap(lr_pipeline()), "all_with_port"),
        "Logistic regression (reduced, no port)": (lambda: SkWrap(lr_pipeline()), "reduced_no_port"),
        "Decision tree depth 4 (all features, no port)": (lambda: SkWrap(DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=RANDOM_STATE)), "all_no_port"),
    }

    results, fitted = [], {}
    for split in ["strat", "time"]:
        tr = (splits[f"{split}_split"] == "train").values
        va = (splits[f"{split}_split"] == "val").values
        for name, (make, fset) in models.items():
            cols = fs[fset]
            Xtr, ytr = df.loc[tr, cols], df.loc[tr, TARGET].values
            Xva, yva = df.loc[va, cols], df.loc[va, TARGET].values
            m = make()
            with Timer() as t:
                m.fit(Xtr, ytr)
            pv, sv = m.predict(Xva), m.score(Xva)
            r = {"split": "stratified" if split == "strat" else "time-ordered", "model": name,
                 "feature_set": fset, "n_features": len(cols), **metrics(yva, pv, sv)}
            r["monday_fpr"] = m.predict(mon[cols]).mean()
            r["monday_rows"] = len(mon)
            r["train_seconds"] = t.s
            r["meets_recall_target"] = r["recall"] >= TARGET_RECALL
            r["meets_fpr_target"] = r["fpr"] <= TARGET_FPR
            r["meets_monday_fpr_target"] = r["monday_fpr"] <= TARGET_FPR
            results.append(r)
            fitted[(split, name)] = (m, cols, Xva, yva, pv, sv)
            print(f"[{split}] {name:42s} P={r['precision']:.4f} R={r['recall']:.4f} "
                  f"F1={r['f1']:.4f} FPR={r['fpr']:.4f} MonFPR={r['monday_fpr']:.4f}")

    res = pd.DataFrame(results)
    save_table(res, "baseline_results")
    rule = fitted[("strat", "Rule-based (duration/packets/payload)")][0]
    save_table(pd.DataFrame([{"rule": rule.describe(), "duration_max_us": rule.d,
                              "fwd_packets_max": rule.n, "fwd_payload_max_bytes": rule.p}]), "rule_definition")
    save_json({"validation_only": True, "test_set_used": False, "rule": rule.describe(),
               "results": results}, "baselines")

    # ---------------- figures (stratified split) --------------------------------------
    lr_name = "Logistic regression (all features, no port)"
    m, cols, Xva, yva, pv, sv = fitted[("strat", lr_name)]

    # Fig 6 confusion matrix
    cm = confusion_matrix(yva, pv)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(cm, cmap="Blues")
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]:,}\n({cm[i, j] / cm[i].sum() * 100:.2f}% of row)",
                    ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    ax.set_xticks([0, 1], ["Predicted BENIGN", "Predicted PortScan"])
    ax.set_yticks([0, 1], ["Actual BENIGN", "Actual PortScan"])
    ax.grid(False)
    ax.set_title("Logistic regression (all features, no port),\nvalidation confusion matrix")
    save_fig(fig, "fig6_confusion_matrix")

    # Fig 7 ROC + PR curves for both LR variants
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for nm, color, ls in [("Logistic regression (all features, no port)", "#2a78d6", "-"),
                          ("Logistic regression (all features + port)", "#eb6834", "--")]:
        _, _, _, y_, _, s_ = fitted[("strat", nm)]
        fpr, tpr, _ = roc_curve(y_, s_)
        pr, rc, _ = precision_recall_curve(y_, s_)
        a1.plot(fpr, tpr, color=color, ls=ls, lw=2, label=f"{nm.split('(')[1][:-1]} (AUC {roc_auc_score(y_, s_):.4f})")
        a2.plot(rc, pr, color=color, ls=ls, lw=2, label=f"{nm.split('(')[1][:-1]} (AP {average_precision_score(y_, s_):.4f})")
    a1.axvline(TARGET_FPR, color="gray", lw=1, ls=":"); a1.text(TARGET_FPR - .001, 0.985, "FPR target 5%", ha="right", color="gray", fontsize=9)
    a1.plot([0, 1], [0, 1], color="lightgray", lw=1)
    a1.set_xlabel("False-positive rate (benign flagged)"); a1.set_ylabel("Recall (scans caught)")
    a1.set_xlim(0, 0.06); a1.set_ylim(0.95, 1.002)
    a1.set_title("ROC curve (zoomed: FPR 0-6%)"); a1.legend(loc="center right", title="Logistic regression")
    a2.axvline(TARGET_RECALL, color="gray", lw=1, ls=":"); a2.text(TARGET_RECALL + .002, 0.985, "Recall target 0.90", color="gray", fontsize=9)
    a2.set_xlabel("Recall (scans caught)"); a2.set_ylabel("Precision")
    a2.set_xlim(0.88, 1.002); a2.set_ylim(0.95, 1.002)
    a2.set_title("Precision-recall curve (zoomed)"); a2.legend(loc="center", title="Logistic regression")
    fig.suptitle("Logistic regression baseline on the validation set (stratified split)",
                 fontweight="bold", fontsize=13)
    save_fig(fig, "fig7_roc_pr_curves")

    # Fig 8 coefficients
    coefs = pd.DataFrame({"feature": cols, "coef": m.m.named_steps["lr"].coef_[0]})
    coefs["abs"] = coefs.coef.abs()
    coefs = coefs.sort_values("abs", ascending=False)
    save_table(coefs.drop(columns="abs"), "lr_coefficients")
    top = coefs.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.feature, top.coef, color=[COLORS["PortScan"] if c > 0 else COLORS["BENIGN"] for c in top.coef], height=0.7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Standardised coefficient (+ pushes toward PortScan, - toward BENIGN)")
    ax.set_title("Top 15 logistic-regression coefficients (all features, no port)")
    ax.grid(axis="y", visible=False)
    save_fig(fig, "fig8_lr_coefficients")

    # ---- Do the columns that the reduced set leaves out carry useful signal? -------------
    tree_m = fitted[("strat", "Decision tree depth 4 (all features, no port)")][0].m
    tree_cols = fitted[("strat", "Decision tree depth 4 (all features, no port)")][1]
    used_in_tree = {tree_cols[i] for i in tree_m.tree_.feature if i >= 0}
    coefs_rank = coefs.reset_index(drop=True)
    coefs_rank["rank"] = np.arange(1, len(coefs_rank) + 1)
    rank = dict(zip(coefs_rank.feature, coefs_rank["rank"]))
    sep = pd.read_csv(MET_DIR.parent / "tables" / "feature_separability.csv").set_index("feature").univariate_auc
    excluded = [c for c in fs["all_no_port"] if c not in fs["reduced_no_port"]]
    chk = pd.DataFrame([{
        "feature": c,
        "why_excluded_from_reduced": ("constant" if df[c].nunique() <= 1 else
                                      "exact copy" if c in __import__("config").DUPLICATE_COLS else "|r| > 0.95"),
        "univariate_auc": sep.get(c, np.nan),
        "lr_all_coef_rank": rank.get(c),
        "lr_all_abs_coef": float(coefs.set_index("feature").coef.get(c, np.nan)),
        "used_in_tree": c in used_in_tree} for c in excluded]).sort_values("lr_all_coef_rank")
    save_table(chk, "excluded_feature_check")
    print(chk.head(10).to_string(index=False))

    # Fig 9 decision tree
    tree, tcols = fitted[("strat", "Decision tree depth 4 (all features, no port)")][:2]
    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(tree.m, feature_names=tcols, class_names=["BENIGN", "PortScan"], filled=True,
              impurity=False, proportion=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title("Depth-4 decision tree baseline (all features, no port), trained on stratified train split\n"
                 "(node colour: orange = mostly BENIGN, blue = mostly PortScan; 'value' = class proportions)",
                 fontsize=15, fontweight="bold")
    save_fig(fig, "fig9_decision_tree")

    # ---------------- error analysis (LR no port, stratified val) ---------------------
    key = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Fwd Packet Length Max",
           "Flow Packets/s", "Init_Win_bytes_forward", "Init_Win_bytes_backward", "Destination Port"]
    va_df = df.loc[(splits["strat_split"] == "val").values, key].copy()
    va_df["outcome"] = np.select(
        [(yva == 1) & (pv == 1), (yva == 0) & (pv == 0), (yva == 1) & (pv == 0), (yva == 0) & (pv == 1)],
        ["TP (scan caught)", "TN (benign passed)", "FN (scan missed)", "FP (false alarm)"], default="")
    ea = va_df.groupby("outcome")[key].median()
    ea.insert(0, "n_flows", va_df.outcome.value_counts())
    save_table(ea.reset_index(), "error_analysis")
    fp_ports = va_df[va_df.outcome == "FP (false alarm)"]["Destination Port"].value_counts().head(5)
    fn_ports = va_df[va_df.outcome == "FN (scan missed)"]["Destination Port"].value_counts().head(5)
    save_table(pd.DataFrame({"false_alarm_top_ports": fp_ports.index.astype(str) + " (" + fp_ports.values.astype(str) + ")",
                             "missed_scan_top_ports": list(fn_ports.index.astype(str) + " (" + fn_ports.values.astype(str) + ")")
                             + [""] * (len(fp_ports) - len(fn_ports))}), "error_ports")
    print(ea.to_string())


if __name__ == "__main__":
    main()
