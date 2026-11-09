"""Step 9 (Deliverable 3): Evaluation figures and error analysis.

Reads the models fitted in step 8 and produces the evaluation evidence for the Status Report:
precision-recall and ROC curves for the three families, confusion matrices on the locked test
split, and a structured error analysis of the false positives and false negatives of the
selected model (representative cases, likely cause, security consequence).

Outputs:
  figures/fig10_model_comparison.png, fig11_confusion_matrices.png, fig12_error_profile.png
  tables/error_analysis_v2.csv, error_cases.csv, error_causes.csv, fn_fp_ports.csv
  metrics/error_analysis_v2.json
"""
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, precision_recall_curve, roc_curve,
                             average_precision_score, roc_auc_score)

from config import PROC_DIR, SPLIT_FILE, MET_DIR, TAB_DIR, TARGET, PORT_COL, COLORS, TARGET_FPR
from utils import save_table, save_json, save_fig, style, select_model

style()
MODEL_MATRIX = PROC_DIR / "model_matrix.parquet"
MODEL_DIR = PROC_DIR / "models"
# Read from the comparison table so a family added in step 08 appears here automatically.
MODEL_COLORS = {"Logistic regression": "#2a78b5", "Random Forest": "#4c9f70",
                "Gradient Boosting": "#e07b1a", "XGBoost": "#8e5ea2"}


def model_names():
    comp = pd.read_csv(TAB_DIR / "model_comparison.csv")
    order = ["Logistic regression", "Random Forest", "Gradient Boosting", "XGBoost"]
    present = list(dict.fromkeys(comp.model))
    return [m for m in order if m in present] + [m for m in present if m not in order]

# Features shown when profiling an individual error case.
CASE_COLS = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
             "Total Length of Fwd Packets", "Total Length of Bwd Packets",
             "SYN Flag Count", "RST Flag Count", "ACK Flag Count",
             "Init_Win_bytes_forward", "Init_Win_bytes_backward", PORT_COL]


def load_all():
    df = pd.read_parquet(MODEL_MATRIX)
    splits = pd.read_parquet(SPLIT_FILE)
    fs = json.load(open(MET_DIR / "feature_sets_v2.json"))
    thr = pd.read_csv(TAB_DIR / "threshold_choice.csv").set_index("model")["threshold"].to_dict()
    models = {m: joblib.load(MODEL_DIR / f"{m.lower().replace(' ', '_')}.joblib")
              for m in model_names()}
    return df, splits, fs, thr, models





def main():
    MODELS = model_names()
    df, splits, fs, thr, models = load_all()
    cols = fs["full"]
    m_val = (splits.strat_split == "val").values
    m_test = (splits.strat_split == "test").values
    yv, yt = df.loc[m_val, TARGET].values, df.loc[m_test, TARGET].values
    Xv, Xt = df.loc[m_val, cols], df.loc[m_test, cols]

    sv = {n: m.predict_proba(Xv)[:, 1] for n, m in models.items()}
    st = {n: m.predict_proba(Xt)[:, 1] for n, m in models.items()}
    best = select_model()
    print(f"selected model: {best}")

    # ---------------- Figure 10: primary comparison -----------------------------------------
    # All three models sit on top of each other on a normal PR plot, so panel (a) counts the
    # errors instead: at this level of performance that is the only thing that separates them.
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.8))

    err = pd.DataFrame([{
        "model": n,
        "FN": int(((yt == 1) & (st[n] < thr[n])).sum()),
        "FP": int(((yt == 0) & (st[n] >= thr[n])).sum()),
    } for n in MODELS])
    x = np.arange(len(err))
    b1 = a1.bar(x - 0.2, err.FN, width=0.4, color="#e07b1a", label="Missed scans (FN)")
    b2 = a1.bar(x + 0.2, err.FP, width=0.4, color="#2a78b5", label="False alarms (FP)")
    for bars in (b1, b2):
        for rect in bars:
            a1.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.4,
                    f"{int(rect.get_height())}", ha="center", fontsize=10, fontweight="bold")
    a1.set_xticks(x, [n.replace(" ", "\n") for n in err.model], fontsize=9)
    a1.set_ylabel("Flows misclassified (of 32,041)")
    a1.set_ylim(0, max(err.FN.max(), err.FP.max()) * 1.35 + 1)
    a1.set_title("(a) Where the models actually differ", fontsize=11)
    a1.legend(fontsize=9, loc="upper left")
    a1.grid(axis="x", visible=False)

    for n in MODELS:
        pr, rc, _ = precision_recall_curve(yt, st[n])
        a2.plot(rc, pr, color=MODEL_COLORS[n], lw=2, alpha=0.85,
                label=f"{n} ({average_precision_score(yt, st[n]):.5f})")
    a2.set_xlim(0.994, 1.0008); a2.set_ylim(0.9975, 1.0004)
    a2.set_xlabel("Recall (share of scans caught)"); a2.set_ylabel("Precision")
    a2.set_title("(b) Precision-recall, zoomed into the corner", fontsize=11)
    a2.legend(loc="lower left", fontsize=8.5, title="PR-AUC", title_fontsize=8.5)

    cm = confusion_matrix(yt, (st[best] >= thr[best]).astype(int))
    a3.imshow(cm, cmap="Blues")
    names = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            a3.text(j, i, f"{names[i][j]}\n{cm[i, j]:,}\n({cm[i, j] / cm[i].sum() * 100:.2f}% of row)",
                    ha="center", va="center", fontsize=9.5,
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    a3.set_xticks([0, 1], ["Predicted BENIGN", "Predicted PortScan"])
    a3.set_yticks([0, 1], ["Actual BENIGN", "Actual PortScan"])
    a3.grid(False); a3.set_title(f"(c) {best}, locked test split")
    fig.suptitle("ScanShield model comparison on the locked test split (32,041 flows)",
                 fontweight="bold", fontsize=13)
    save_fig(fig, "fig10_model_comparison")

    # ---------------- Figure 11: confusion matrices for all three -------------------------
    fig, axes = plt.subplots(1, len(MODELS), figsize=(3.9 * len(MODELS), 4.0))
    for ax, n in zip(axes, MODELS):
        c = confusion_matrix(yt, (st[n] >= thr[n]).astype(int))
        ax.imshow(c, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{c[i, j]:,}", ha="center", va="center", fontsize=12,
                        color="white" if c[i, j] > c.max() * 0.5 else "black")
        ax.set_xticks([0, 1], ["Pred. BENIGN", "Pred. Scan"], fontsize=9)
        ax.set_yticks([0, 1], ["BENIGN", "Scan"], fontsize=9)
        ax.grid(False)
        ax.set_title(f"{n}\nFP={c[0,1]}  FN={c[1,0]}", fontsize=10)
    fig.suptitle("Confusion matrices at each model's frozen operating threshold (test split)",
                 fontweight="bold", fontsize=13)
    save_fig(fig, "fig11_confusion_matrices")

    # ---------------- Error analysis for the selected model (validation + test) -----------
    rows = []
    for part, mask, y, s in [("validation", m_val, yv, sv[best]), ("test (locked)", m_test, yt, st[best])]:
        pred = (s >= thr[best]).astype(int)
        sub = df.loc[mask].copy()
        sub["outcome"] = np.select(
            [(y == 1) & (pred == 1), (y == 0) & (pred == 0), (y == 1) & (pred == 0), (y == 0) & (pred == 1)],
            ["TP (scan caught)", "TN (benign passed)", "FN (scan missed)", "FP (false alarm)"], default="")
        sub["split"] = part
        sub["model_score"] = s
        rows.append(sub)
    err = pd.concat(rows)

    prof = (err.groupby(["split", "outcome"])[CASE_COLS].median().reset_index())
    counts = err.groupby(["split", "outcome"]).size().rename("n_flows").reset_index()
    prof = counts.merge(prof, on=["split", "outcome"])
    save_table(prof, "error_analysis_v2")
    print(prof.to_string(index=False))

    # representative individual cases (worst-scoring misses, highest-scoring false alarms)
    cases = []
    for part in ["validation", "test (locked)"]:
        e = err[err.split == part]
        fn = e[e.outcome == "FN (scan missed)"].nsmallest(4, "model_score")
        fp = e[e.outcome == "FP (false alarm)"].nlargest(4, "model_score")
        for kind, sel in [("False negative (scan missed)", fn), ("False positive (false alarm)", fp)]:
            for _, r in sel.iterrows():
                cases.append({"split": part, "error_type": kind, "model_score": round(r.model_score, 4),
                              **{c: r[c] for c in CASE_COLS}})
    save_table(pd.DataFrame(cases), "error_cases")

    # grouped causes
    e = err[err.split == "test (locked)"]
    fp, fn = e[e.outcome == "FP (false alarm)"], e[e.outcome == "FN (scan missed)"]

    def share(frame, cond):
        return float(cond.mean()) if len(frame) else 0.0

    causes = [
        {"error_type": "False positive", "pattern": "Benign flow that received no reply at all",
         "n": int(fp["no_response"].sum()), "share_of_errors": share(fp, fp["no_response"] == 1)},
        {"error_type": "False positive", "pattern": "Benign flow carrying zero payload bytes",
         "n": int(fp["is_empty_payload"].sum()), "share_of_errors": share(fp, fp["is_empty_payload"] == 1)},
        {"error_type": "False positive", "pattern": "Benign flow shorter than 1 ms",
         "n": int(fp["is_short_flow"].sum()), "share_of_errors": share(fp, fp["is_short_flow"] == 1)},
        {"error_type": "False negative", "pattern": "Scan flow that did get a reply packet",
         "n": int((fn["Total Backward Packets"] > 0).sum()),
         "share_of_errors": share(fn, fn["Total Backward Packets"] > 0)},
        {"error_type": "False negative", "pattern": "Scan flow carrying payload bytes",
         "n": int((fn["total_bytes"] > 0).sum()), "share_of_errors": share(fn, fn["total_bytes"] > 0)},
        {"error_type": "False negative", "pattern": "Scan flow aimed at a common service port",
         "n": int(fn["is_common_service_port"].sum()),
         "share_of_errors": share(fn, fn["is_common_service_port"] == 1)},
        {"error_type": "False negative", "pattern": "Scan flow lasting longer than 1 ms",
         "n": int((fn["is_short_flow"] == 0).sum()), "share_of_errors": share(fn, fn["is_short_flow"] == 0)},
    ]
    save_table(pd.DataFrame(causes), "error_causes")

    fp_ports = fp[PORT_COL].value_counts().head(6)
    fn_ports = fn[PORT_COL].value_counts().head(6)
    save_table(pd.DataFrame({
        "rank": range(1, max(len(fp_ports), len(fn_ports)) + 1),
        "false_alarm_port": list(fp_ports.index) + [None] * (max(len(fp_ports), len(fn_ports)) - len(fp_ports)),
        "false_alarm_n": list(fp_ports.values) + [None] * (max(len(fp_ports), len(fn_ports)) - len(fp_ports)),
        "missed_scan_port": list(fn_ports.index) + [None] * (max(len(fp_ports), len(fn_ports)) - len(fn_ports)),
        "missed_scan_n": list(fn_ports.values) + [None] * (max(len(fp_ports), len(fn_ports)) - len(fn_ports)),
    }), "fn_fp_ports")

    # ---------------- Figure 12: what the errors look like --------------------------------
    e_all = err[err.split == "test (locked)"]
    panels = [("Flow Duration", "Flow duration (microseconds, log scale)", True),
              ("total_bytes", "Total payload bytes (log 1+x)", True),
              ("Total Backward Packets", "Packets returned by the responder", False),
              ("model_score", "Model score (probability of PortScan)", False)]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    order = ["TN (benign passed)", "FP (false alarm)", "FN (scan missed)", "TP (scan caught)"]
    pal = {"TN (benign passed)": "#9ecae1", "FP (false alarm)": "#2a78b5",
           "FN (scan missed)": "#fdae61", "TP (scan caught)": "#e07b1a"}
    for ax, (col, label, logx) in zip(axes, panels):
        data, labels, colors = [], [], []
        for o in order:
            v = e_all.loc[e_all.outcome == o, col].astype(float).values
            if len(v) == 0:
                continue
            data.append(np.log10(v + 1) if logx else v)
            labels.append(f"{o.split(' (')[0]}\n(n={len(v):,})")
            colors.append(pal[o])
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False, widths=0.6)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.85)
        ax.set_ylabel(("log10(1 + " + label.split(" (")[0] + ")") if logx else label, fontsize=9.5)
        ax.set_title(label, fontsize=10.5)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"What {best} gets wrong on the locked test split: errors sit between the two classes",
                 fontweight="bold", fontsize=13)
    save_fig(fig, "fig12_error_profile")

    save_json({"selected_model": best, "threshold": thr[best],
               "test_FP": int((e_all.outcome == "FP (false alarm)").sum()),
               "test_FN": int((e_all.outcome == "FN (scan missed)").sum()),
               "causes": causes}, "error_analysis_v2")
    print(f"\nFP={int((e_all.outcome=='FP (false alarm)').sum())} "
          f"FN={int((e_all.outcome=='FN (scan missed)').sum())} on test for {best}")


if __name__ == "__main__":
    main()
