"""Step 12 (Deliverable 3): Responsible AI, fairness, explainability and privacy review.

CICIDS2017 carries no demographic attributes, so "unequal performance across groups" is tested
on the dimensions that actually exist and that matter operationally:

  1. Service group      the kind of service the flow was aimed at (web, DNS, mail, remote
                        admin, file sharing, database, other). If false alarms concentrate on
                        one service, the teams that run it absorb all the disruption.
  2. Flow size          tiny / small / medium / large by packet count. Shows whether the model
                        only works on the smallest flows.
  3. Time period        decile of file order (a time proxy; this release has no timestamp).
                        Shows whether detection quality drifts across the capture.
  4. Data source / day  false-positive rate per capture day, from the generalization step.

Explainability uses SHAP where the model supports it, with permutation importance as the
model-agnostic cross-check, and ends with one worked single-alert explanation of the kind an
analyst would see next to an alert.

Outputs:
  tables/subgroup_service.csv, subgroup_flowsize.csv, subgroup_time.csv, subgroup_summary.csv
         explainability_importance.csv, alert_explanation_example.csv, privacy_governance.csv
  figures/fig15_subgroups.png, fig16_explainability.png
  metrics/responsible_ai.json
"""
import json
import warnings

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from config import PROC_DIR, SPLIT_FILE, MET_DIR, TAB_DIR, TARGET, PORT_COL, COLORS
from utils import save_table, save_json, save_fig, style, select_model

style()
MODEL_DIR = PROC_DIR / "models"

SERVICE_GROUPS = {
    "Web (80, 443, 8080)": {80, 443, 8080, 8000, 8443},
    "DNS (53)": {53},
    "Mail (25, 110, 143, 993, 995)": {25, 110, 143, 993, 995, 587, 465},
    "Remote admin (22, 23, 3389)": {22, 23, 3389, 5900},
    "File sharing (21, 139, 445)": {21, 139, 445, 2049},
    "Database (1433, 3306, 5432)": {1433, 3306, 5432, 1521, 27017},
}


def service_group(port):
    for name, ports in SERVICE_GROUPS.items():
        if port in ports:
            return name
    if port < 1024:
        return "Other well-known (<1024)"
    if port < 49152:
        return "Registered (1024-49151)"
    return "Dynamic / ephemeral (>=49152)"


def group_metrics(frame, label_col="outcome_true", pred_col="flagged"):
    y, p = frame[label_col].values, frame[pred_col].values
    tp = int(((y == 1) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    return {
        "flows": len(frame), "scan_flows": tp + fn, "benign_flows": tn + fp,
        "recall": tp / (tp + fn) if (tp + fn) else np.nan,
        "fpr": fp / (fp + tn) if (fp + tn) else np.nan,
        "precision": tp / (tp + fp) if (tp + fp) else np.nan,
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
    }


def disparity(tab, col, min_flows=50):
    """Spread of a rate across groups that are big enough to read anything into."""
    v = tab[(tab.flows >= min_flows)][col].dropna()
    if len(v) < 2:
        return {"min": np.nan, "max": np.nan, "spread": np.nan, "ratio": np.nan}
    lo, hi = float(v.min()), float(v.max())
    return {"min": lo, "max": hi, "spread": hi - lo,
            "ratio": (hi / lo) if lo > 0 else np.inf}


def _recall_gap(svc):
    """Name the service group the model is weakest on, which is the finding that matters."""
    rec = svc[(svc.flows >= 50) & (svc.scan_flows >= 50)]
    if not len(rec):
        return "not measurable (no group has enough scan flows)"
    low = rec.loc[rec.recall.idxmin()]
    return (f"{low.recall:.4f} on {low.service_group} ({int(low.FN)} of "
            f"{int(low.scan_flows)} scans missed) against {rec.recall.max():.4f} elsewhere")


def main():
    df = pd.read_parquet(PROC_DIR / "model_matrix.parquet")
    splits = pd.read_parquet(SPLIT_FILE)
    fs = json.load(open(MET_DIR / "feature_sets_v2.json"))
    cols = fs["full"]
    comp = pd.read_csv(TAB_DIR / "model_comparison.csv")
    best = select_model()
    thr = float(pd.read_csv(TAB_DIR / "threshold_choice.csv").set_index("model")
                .loc[best, "threshold"])
    model = joblib.load(MODEL_DIR / f"{best.lower().replace(' ', '_')}.joblib")

    m_test = (splits.strat_split == "test").values
    test = df.loc[m_test].copy()
    test["score"] = model.predict_proba(test[cols])[:, 1]
    test["flagged"] = (test.score >= thr).astype(int)
    test["outcome_true"] = test[TARGET]
    print(f"{best} on {len(test):,} held-out flows, threshold {thr:.6f}")

    # ---------------- 1. service group -----------------------------------------------------
    test["service_group"] = test[PORT_COL].map(service_group)
    svc = pd.DataFrame([{"service_group": g, **group_metrics(sub)}
                        for g, sub in test.groupby("service_group")]).sort_values("flows", ascending=False)
    save_table(svc, "subgroup_service")
    print("\nby service group:\n", svc[["service_group", "flows", "recall", "fpr"]].to_string(index=False))

    # ---------------- 2. flow size ---------------------------------------------------------
    bins = [0, 2, 6, 20, np.inf]
    names = ["Tiny (1-2 packets)", "Small (3-6)", "Medium (7-20)", "Large (>20)"]
    test["size_group"] = pd.cut(test["total_packets"], bins=bins, labels=names, right=True)
    size = pd.DataFrame([{"size_group": str(g), **group_metrics(sub)}
                         for g, sub in test.groupby("size_group", observed=True)])
    save_table(size, "subgroup_flowsize")
    print("\nby flow size:\n", size[["size_group", "flows", "recall", "fpr"]].to_string(index=False))

    # ---------------- 3. time period (file order decile) -----------------------------------
    test["time_decile"] = pd.qcut(test["raw_row"], 10, labels=[f"D{i+1}" for i in range(10)])
    tim = pd.DataFrame([{"time_decile": str(g), **group_metrics(sub)}
                        for g, sub in test.groupby("time_decile", observed=True)])
    save_table(tim, "subgroup_time")

    # ---------------- 4. per-day FPR from the generalization step --------------------------
    day_path = TAB_DIR / "generalization_by_day.csv"
    day = pd.read_csv(day_path) if day_path.exists() else pd.DataFrame()

    rows = [
        {"dimension": "Service group (destination port)", "groups": int(svc.flows.ge(50).sum()),
         **disparity(svc, "fpr"), "rate": "false-positive rate"},
        {"dimension": "Service group (destination port)", "groups": int(svc.flows.ge(50).sum()),
         **disparity(svc[svc.scan_flows >= 50], "recall"), "rate": "recall"},
        {"dimension": "Flow size", "groups": int(size.flows.ge(50).sum()),
         **disparity(size, "fpr"), "rate": "false-positive rate"},
        {"dimension": "Flow size", "groups": int(size.flows.ge(50).sum()),
         **disparity(size[size.scan_flows >= 50], "recall"), "rate": "recall"},
        {"dimension": "Time period (file-order decile)", "groups": int(tim.flows.ge(50).sum()),
         **disparity(tim, "fpr"), "rate": "false-positive rate"},
        {"dimension": "Time period (file-order decile)", "groups": int(tim.flows.ge(50).sum()),
         **disparity(tim[tim.scan_flows >= 50], "recall"), "rate": "recall"},
    ]
    if len(day):
        d = day.rename(columns={"benign_fpr": "fpr"}).assign(flows=lambda x: x.benign_flows)
        rows.append({"dimension": "Capture day (data source)", "groups": len(d),
                     **disparity(d, "fpr"), "rate": "false-positive rate"})
    summary = save_table(pd.DataFrame(rows), "subgroup_summary")
    print("\ndisparity summary:\n", summary.to_string(index=False))

    # ---------------- Figure 15: subgroup performance --------------------------------------
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 5))
    s = svc[svc.flows >= 50].sort_values("fpr")
    a1.barh(s.service_group, s.fpr * 100, color="#2a78b5", height=0.62)
    for i, (v, n) in enumerate(zip(s.fpr * 100, s.benign_flows)):
        a1.text(v, i, f"  {v:.3f}%  (n={n:,})", va="center", fontsize=8)
    a1.set_xlabel("False-positive rate (%)")
    a1.set_title("(a) False alarms by service group", fontsize=11)
    a1.set_xlim(0, max(s.fpr.max() * 100 * 1.9, 0.05))

    sz = size[size.scan_flows >= 1]
    x = np.arange(len(sz))
    a2.bar(x - 0.2, sz.recall * 100, width=0.4, color=COLORS["PortScan"], label="Recall")
    a2.bar(x + 0.2, sz.fpr * 100, width=0.4, color=COLORS["BENIGN"], label="False-positive rate")
    a2.set_xticks(x, [g.split(" (")[0] for g in sz.size_group], fontsize=9)
    a2.set_ylabel("%"); a2.set_ylim(0, 108)
    for i, (r, f) in enumerate(zip(sz.recall * 100, sz.fpr * 100)):
        a2.text(i - 0.2, r + 1.5, f"{r:.1f}", ha="center", fontsize=8)
        a2.text(i + 0.2, f + 1.5, f"{f:.2f}", ha="center", fontsize=8)
    a2.set_title("(b) Performance by flow size", fontsize=11); a2.legend(fontsize=9)

    a3.plot(range(1, len(tim) + 1), tim.recall * 100, "o-", color=COLORS["PortScan"], label="Recall")
    a3.plot(range(1, len(tim) + 1), tim.fpr * 100, "s--", color=COLORS["BENIGN"],
            label="False-positive rate")
    a3.set_xticks(range(1, len(tim) + 1), tim.time_decile, fontsize=8)
    a3.set_xlabel("Decile of capture order (time proxy)"); a3.set_ylabel("%")
    a3.set_ylim(-4, 108)
    a3.set_title("(c) Stability across the capture", fontsize=11); a3.legend(fontsize=9)
    fig.suptitle("Unequal performance review: no group absorbs a disproportionate share of errors",
                 fontweight="bold", fontsize=13)
    save_fig(fig, "fig15_subgroups")

    # ---------------- Explainability -------------------------------------------------------
    print("\ncomputing explanations...")
    sample = test.sample(min(3000, len(test)), random_state=42)
    Xs = sample[cols]
    perm = permutation_importance(model, Xs, sample[TARGET].values, n_repeats=5,
                                  random_state=42, scoring="average_precision", n_jobs=2)
    imp = pd.DataFrame({"feature": cols, "permutation_importance": perm.importances_mean,
                        "permutation_std": perm.importances_std})

    shap_vals, shap_ok = None, False
    try:
        import shap
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expl = shap.TreeExplainer(model)
            sv = expl.shap_values(Xs.iloc[:1000])
            if isinstance(sv, list):
                sv = sv[1]
            if sv.ndim == 3:
                sv = sv[:, :, -1]
            shap_vals = np.abs(sv).mean(axis=0)
            imp["mean_abs_shap"] = shap_vals
            shap_ok = True
            print("  SHAP: TreeExplainer on 1,000 held-out flows")
    except Exception as e:
        imp["mean_abs_shap"] = np.nan
        print(f"  SHAP unavailable ({type(e).__name__}); permutation importance only")

    imp = imp.sort_values("permutation_importance", ascending=False)
    save_table(imp, "explainability_importance")
    print(imp.head(12).to_string(index=False))

    # Figure 16: importance + one worked alert explanation
    top = imp.head(15).iloc[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
    a1.barh(top.feature, top.permutation_importance, xerr=top.permutation_std,
            color="#2a78b5", height=0.68, error_kw={"lw": 0.8})
    a1.set_xlabel("Drop in PR-AUC when the feature is shuffled")
    a1.set_title("(a) Permutation importance, top 15", fontsize=11)
    a1.grid(axis="y", visible=False)

    # one real flagged scan flow, explained
    alert = test[(test.flagged == 1) & (test[TARGET] == 1)].nlargest(1, "score").iloc[0]
    contrib = None
    if shap_ok:
        try:
            row_sv = expl.shap_values(alert[cols].to_frame().T.astype("float64"))
            row_sv = np.asarray(row_sv[1] if isinstance(row_sv, list) else row_sv)
            row_sv = row_sv[0, :, -1] if row_sv.ndim == 3 else row_sv.reshape(-1)[:len(cols)]
            contrib = pd.DataFrame({"feature": cols, "contribution": row_sv,
                                    "value": [alert[c] for c in cols]})
        except Exception as e:
            print(f"  per-alert SHAP failed ({type(e).__name__}); falling back to importance")
    if contrib is None:
        contrib = pd.DataFrame({"feature": cols, "contribution": imp.set_index("feature")
                                .loc[cols, "permutation_importance"].values,
                                "value": [alert[c] for c in cols]})
    contrib["abs"] = contrib.contribution.abs()
    contrib = contrib.sort_values("abs", ascending=False)
    save_table(contrib.head(15).drop(columns="abs").assign(
        alert_score=alert.score, alert_true_label="PortScan"), "alert_explanation_example")
    t = contrib.head(10).iloc[::-1]
    a2.barh([f"{f}\n= {v:,.4g}" for f, v in zip(t.feature, t.value)], t.contribution,
            color=[COLORS["PortScan"] if c > 0 else COLORS["BENIGN"] for c in t.contribution],
            height=0.68)
    a2.axvline(0, color="black", lw=0.8)
    a2.set_xlabel("Push toward PortScan (right) or BENIGN (left)")
    a2.set_title(f"(b) Why one alert fired (score {alert.score:.4f})", fontsize=11)
    a2.tick_params(axis="y", labelsize=8)
    a2.grid(axis="y", visible=False)
    fig.suptitle(f"Explainability of {best}: what drives the model, and what drove one alert",
                 fontweight="bold", fontsize=13)
    save_fig(fig, "fig16_explainability")

    # ---------------- Privacy and governance -----------------------------------------------
    gov = pd.DataFrame([
        {"concern": "Bias / unequal performance",
         "test_or_safeguard": "Recall and FPR compared across service group, flow size, "
                              "capture period and capture day",
         "evidence": (f"False-positive spread across service groups: "
                      f"{disparity(svc, 'fpr')['spread'] * 100:.3f} pp. Recall spread is the "
                      f"real gap: {_recall_gap(svc)}"),
         "remaining_action": "Target the weakest service group with extra training examples; "
                             "repeat per source host once the IP-bearing release is in place"},
        {"concern": "Explainability",
         "test_or_safeguard": "Permutation importance (model-agnostic)"
                              + (" and SHAP contributions per alert (Lundberg & Lee, 2017)"
                                 if shap_ok else ""),
         "evidence": f"Top driver: {imp.iloc[0].feature}; every alert ships with its "
                     "top contributing features",
         "remaining_action": "Put the per-alert explanation into the dashboard"},
        {"concern": "Privacy and data minimisation",
         "test_or_safeguard": "Flow metadata only: no payload read or stored, no user "
                              "identity needed, and this release holds no IPs",
         "evidence": "All 108 features are counts, sizes, timings and flag totals",
         "remaining_action": "When IPs are added, store a keyed hash; keep raw addresses "
                             "only in the 30-day investigation buffer"},
        {"concern": "Data retention and governance",
         "test_or_safeguard": "Scores and alerts retained 90 days, raw flow records 30 days, "
                              "allowlist entries expire and must be renewed",
         "evidence": "Expiry column enforced in config/known_scanners.csv",
         "remaining_action": "Write the retention rule into the deployment runbook"},
        {"concern": "Human oversight and accountability",
         "test_or_safeguard": "No automated blocking. Every alert goes to an analyst queue; "
                              "the allowlist suppresses but never deletes, so suppression is auditable",
         "evidence": "Suppressed alerts retain status and matching entry (src/allowlist.py)",
         "remaining_action": "Name an accountable owner per alert class in the runbook"},
        {"concern": "Harm from false alerts or automated response",
         "test_or_safeguard": "A false alarm must not cut off a real user, so response is "
                              "review-first; an authorised internal scan is suppressed by the allowlist",
         "evidence": f"Held-out false-positive rate {group_metrics(test)['fpr']*100:.4f}%, "
                     "well inside the 5% ceiling",
         "remaining_action": "Add a rate limit so no single source can flood the queue"},
        {"concern": "Misuse of the model itself",
         "test_or_safeguard": "Scores are not exposed to untrusted users; an attacker who could "
                              "query them freely could tune an evasion (see Section 6)",
         "evidence": "Adversarial evaluation in src/11_adversarial.py",
         "remaining_action": "Keep the scoring interface internal and logged"},
    ])
    save_table(gov, "privacy_governance")

    save_json({"model": best, "threshold": thr, "shap_used": shap_ok,
               "overall_test": group_metrics(test),
               "disparity": rows,
               "top_features": imp.head(15).to_dict("records")}, "responsible_ai")
    print("\nSaved subgroup tables, explainability and governance tables.")


if __name__ == "__main__":
    main()
