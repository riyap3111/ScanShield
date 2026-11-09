"""Step 13 (Deliverable 3): Monitoring, feedback and response workflow.

Turns the detector into something that can be watched in production:

  Drift        population stability index (PSI) for every feature, training distribution vs
               each unseen capture day. PSI < 0.10 stable, 0.10-0.25 watch, > 0.25 investigate.
  Alert volume alerts per day at the frozen threshold, which is the analyst workload the
               deployment has to absorb.
  Confidence   the score distribution of alerts, used to set the review tiers.
  Triggers     the concrete rules that send the model back for review or retraining.

Outputs:
  tables/drift_psi.csv, drift_by_day.csv, alert_volume.csv, monitoring_plan.csv,
         response_workflow.csv, logging_schema.csv
  figures/fig17_monitoring.png
  metrics/monitoring.json
"""
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from config import PROC_DIR, SPLIT_FILE, MET_DIR, TAB_DIR, TARGET, LABEL_COL
from utils import save_table, save_json, save_fig, style, select_model

style()
SAMPLE_DIR = PROC_DIR / "external_samples"
PSI_WATCH, PSI_INVESTIGATE = 0.10, 0.25


def psi(expected, actual, bins=10):
    """Population stability index between a reference and a live sample of one feature."""
    e = np.asarray(expected, dtype="float64")
    a = np.asarray(actual, dtype="float64")
    e, a = e[np.isfinite(e)], a[np.isfinite(a)]
    if len(e) < 10 or len(a) < 10:
        return np.nan
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, bins=edges)[0] / len(e)
    pa = np.histogram(a, bins=edges)[0] / len(a)
    eps = 1e-6
    pe, pa = np.clip(pe, eps, None), np.clip(pa, eps, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def main():
    if not SAMPLE_DIR.exists() or not list(SAMPLE_DIR.glob("*.parquet")):
        print(f"No cached day samples in {SAMPLE_DIR}; run 10_generalization.py first.")
        return

    df = pd.read_parquet(PROC_DIR / "model_matrix.parquet")
    splits = pd.read_parquet(SPLIT_FILE)
    fs = json.load(open(MET_DIR / "feature_sets_v2.json"))
    cols = fs["full"]
    comp = pd.read_csv(TAB_DIR / "model_comparison.csv")
    best = select_model()
    thr = float(pd.read_csv(TAB_DIR / "threshold_choice.csv").set_index("model")
                .loc[best, "threshold"])

    ref = df.loc[(splits.strat_split == "train").values]
    ref_benign = ref[ref[TARGET] == 0]
    print(f"reference: {len(ref_benign):,} benign training flows")

    # Whole-day alert totals come from the generalization step, which scored every flow.
    gen_day = pd.read_csv(TAB_DIR / "generalization_by_day.csv").set_index("day")

    # ---------------- drift + alert volume per unseen day ----------------------------------
    psi_rows, day_rows, score_samples = [], [], {}
    for path in sorted(SAMPLE_DIR.glob("*.parquet")):
        live = pd.read_parquet(path)
        day = next((d for d in gen_day.index
                    if d.lower().replace(" ", "_") == path.stem), path.stem)
        live_benign = live[live[LABEL_COL] == "BENIGN"]

        for c in cols:
            psi_rows.append({"day": day, "feature": c,
                             "psi": psi(ref_benign[c].values, live_benign[c].values)})

        flagged = live.flagged.values.astype(bool)
        score_samples[day] = live.score.values[flagged][:20000]
        row = gen_day.loc[day] if day in gen_day.index else None
        n_scored = int(row.rows_after_cleaning) if row is not None else len(live)
        alerts = (int(row.benign_flagged) if row is not None else int(flagged.sum()))
        # benign_flagged counts benign only; scale the sampled flag rate for the whole-day total
        total_alerts = int(round(flagged.mean() * n_scored))
        day_rows.append({
            "day": day, "flows_scored": n_scored,
            "benign_flows": int(row.benign_flows) if row is not None else len(live_benign),
            "benign_alerts": alerts,
            "alerts": total_alerts,
            "alerts_per_10k_flows": float(flagged.mean() * 10_000),
            "median_alert_score": float(np.median(live.score.values[flagged])) if flagged.any() else np.nan,
            "sampled_flows": len(live),
            "max_feature_psi": float(np.nanmax([r["psi"] for r in psi_rows if r["day"] == day])),
        })
        print(f"  {day:15s} alerts ~{total_alerts:>6,}  max PSI {day_rows[-1]['max_feature_psi']:.3f}")
        del live

    if not day_rows:
        print("No cached day samples found; monitoring step needs them.")
        return

    psi_tab = pd.DataFrame(psi_rows)
    worst = (psi_tab.groupby("feature").psi.max().sort_values(ascending=False)
             .reset_index().rename(columns={"psi": "max_psi_across_days"}))
    worst["status"] = np.select(
        [worst.max_psi_across_days > PSI_INVESTIGATE, worst.max_psi_across_days > PSI_WATCH],
        ["investigate", "watch"], default="stable")
    save_table(worst, "drift_psi")
    save_table(psi_tab.pivot(index="feature", columns="day", values="psi").reset_index(),
               "drift_by_day")
    vol = save_table(pd.DataFrame(day_rows), "alert_volume")
    print(f"\nfeatures above the investigate line: "
          f"{int((worst.max_psi_across_days > PSI_INVESTIGATE).sum())} of {len(worst)}")

    # ---------------- monitoring plan -------------------------------------------------------
    med_alerts = float(np.median(vol.alerts))
    p90_alerts = float(np.quantile(vol.alerts, 0.9))
    max_alerts = float(vol.alerts.max())
    quiet_days = int((vol.alerts == 0).sum())
    # Most unseen days raise nothing at all, so a multiple-of-the-median rule degenerates.
    # Band on the busiest observed day instead, and treat any noise on a historically silent
    # day as its own signal.
    if med_alerts == 0:
        volume_rule = (f">{3 * max(max_alerts, 1):,.0f} alerts/day (3x the busiest of "
                       f"{len(vol)} unseen days), or any alert from a source and port that "
                       f"normally produce none")
    else:
        volume_rule = (f"Daily alerts outside 0.5x-3x the observed baseline (median "
                       f"{med_alerts:,.0f}/day, 90th pct {p90_alerts:,.0f})")
    plan = pd.DataFrame([
        {"signal": "Feature drift (PSI vs training benign traffic)",
         "threshold_or_trigger": f"Any feature PSI > {PSI_INVESTIGATE} for 2 consecutive days, "
                                 f"or > {PSI_WATCH} for 7 days",
         "owner_and_response": "ML owner (Ojas) re-checks the feature against the capture "
                               "pipeline, then decides retrain or roll back",
         "feedback_use": "Drifting features are the first candidates for the next refit"},
        {"signal": "Alert volume",
         "threshold_or_trigger": volume_rule,
         "owner_and_response": "SOC duty analyst triages; a spike is either a real scan "
                               "campaign or a broken sensor, and both need an answer the same day",
         "feedback_use": "Volume baseline is recomputed monthly"},
        {"signal": "Score distribution of alerts",
         "threshold_or_trigger": "Median alert score drops below 0.90, or the share of alerts "
                                 "scoring 0.50-0.70 doubles",
         "owner_and_response": "ML owner reviews calibration; borderline alerts mean the "
                               "boundary has moved",
         "feedback_use": "Triggers threshold re-selection on fresh labelled data"},
        {"signal": "Analyst-confirmed precision",
         "threshold_or_trigger": "Confirmed-true rate on sampled alerts falls below 90% "
                                 "over a rolling 2 weeks",
         "owner_and_response": "SOC lead and ML owner review the false alarms together",
         "feedback_use": "Confirmed labels go straight into the retraining pool"},
        {"signal": "Missed scans found after the fact",
         "threshold_or_trigger": "Any scan confirmed by another control that ScanShield did "
                                 "not flag",
         "owner_and_response": "Treated as an incident review: the flow is replayed against "
                               "the model and added to the regression set",
         "feedback_use": "Becomes a permanent test case the next model must pass"},
        {"signal": "Allowlist suppressions",
         "threshold_or_trigger": "Suppression share above 25% of alerts, or any entry past "
                                 "its expiry date",
         "owner_and_response": "Security team re-approves or removes the entry",
         "feedback_use": "Keeps the allowlist from quietly becoming a blind spot"},
        {"signal": "Model freshness",
         "threshold_or_trigger": "Scheduled review every 90 days regardless of other signals",
         "owner_and_response": "Both owners re-run the full pipeline on recent traffic",
         "feedback_use": "Retrain on the accumulated confirmed labels"},
    ])
    save_table(plan, "monitoring_plan")

    workflow = pd.DataFrame([
        {"step": 1, "stage": "Score",
         "action": "Every completed flow is scored; score, model version, feature-set version "
                   "and threshold are written to the alert log",
         "automation": "Automatic"},
        {"step": 2, "stage": "Allowlist check",
         "action": "Alerts matching an approved scanner entry on source, port, schedule and "
                   "expiry are marked suppressed, never deleted",
         "automation": "Automatic, fully audited"},
        {"step": 3, "stage": "Tier",
         "action": "Score >= 0.99 high, 0.90-0.99 medium, threshold-0.90 low; low-tier alerts "
                   "are grouped by source rather than raised one by one",
         "automation": "Automatic"},
        {"step": 4, "stage": "Correlate",
         "action": "Alerts are grouped per source host over 5 min, 1 h and 24 h windows; a "
                   "source crossing 50 distinct ports with over 80% unanswered is escalated "
                   "even if each flow scored low",
         "automation": "Automatic (src/slow_scan.py)"},
        {"step": 5, "stage": "Human review",
         "action": "An analyst confirms or rejects. No automatic blocking at any tier: a false "
                   "alarm must not be able to cut off a real user",
         "automation": "Human decision, required"},
        {"step": 6, "stage": "Respond",
         "action": "Confirmed external scan: rate-limit the source at the perimeter and open "
                   "an incident. Confirmed internal scan: contact the owner and, if it is "
                   "authorised testing (Scarfone et al., 2008), add a time-boxed allowlist entry",
         "automation": "Human-initiated"},
        {"step": 7, "stage": "Feed back",
         "action": "Every analyst verdict is stored with the flow and the score, forming the "
                   "labelled pool the next model is trained and tested on",
         "automation": "Automatic capture, human label"},
    ])
    save_table(workflow, "response_workflow")

    save_table(pd.DataFrame([
        {"field": "flow_id", "purpose": "Join the alert back to the raw flow record"},
        {"field": "scored_at", "purpose": "When the model saw it, for latency monitoring"},
        {"field": "model_version, feature_set_version, threshold",
         "purpose": "Reproduce exactly which model made the call"},
        {"field": "score", "purpose": "Confidence, drives tiering and calibration checks"},
        {"field": "top_features", "purpose": "Per-alert explanation shown to the analyst"},
        {"field": "allowlist_status, allowlist_entry",
         "purpose": "Whether it was suppressed and by which approved entry"},
        {"field": "analyst_verdict, verdict_at, analyst_id",
         "purpose": "Ground-truth feedback and accountability"},
        {"field": "source_port_count_24h, unanswered_ratio_24h",
         "purpose": "Slow-scan context that per-flow scoring cannot see"},
    ]), "logging_schema")

    # ---------------- Figure 17 --------------------------------------------------------------
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 5))
    top = worst.head(12).iloc[::-1]
    bar_colors = ["#c0392b" if s == "investigate" else "#e8a33d" if s == "watch" else "#2a78b5"
                  for s in top.status]
    a1.barh(top.feature, top.max_psi_across_days, color=bar_colors, height=0.68)
    a1.axvline(PSI_WATCH, color="#e8a33d", ls="--", lw=1)
    a1.axvline(PSI_INVESTIGATE, color="#c0392b", ls="--", lw=1)
    a1.text(PSI_INVESTIGATE, -0.9, " investigate", color="#c0392b", fontsize=8)
    a1.set_xlabel("Max PSI across unseen days")
    a1.set_title("(a) Most-drifted features", fontsize=11)
    a1.tick_params(axis="y", labelsize=8); a1.grid(axis="y", visible=False)

    v = vol.sort_values("alerts_per_10k_flows")
    a2.barh(v.day, v.alerts_per_10k_flows, color="#2a78b5", height=0.62)
    for i, (r, n) in enumerate(zip(v.alerts_per_10k_flows, v.alerts)):
        a2.text(r, i, f"  {r:.1f}  ({n:,} alerts)", va="center", fontsize=8)
    a2.set_xlabel("Alerts per 10,000 flows")
    a2.set_title("(b) Alert volume baseline per day", fontsize=11)
    a2.set_xlim(0, max(v.alerts_per_10k_flows.max() * 1.6, 1))

    allsc = np.concatenate([s for s in score_samples.values() if len(s)]) if score_samples else np.array([])
    if len(allsc):
        a3.hist(allsc, bins=40, color="#e07b1a", alpha=0.85)
        a3.axvline(0.99, color="#c0392b", ls="--", lw=1.2)
        a3.text(0.99, a3.get_ylim()[1] * 0.92, " high tier", color="#c0392b", fontsize=8.5, ha="right")
        a3.axvline(0.90, color="#555", ls=":", lw=1.2)
        a3.text(0.90, a3.get_ylim()[1] * 0.80, " medium tier", color="#555", fontsize=8.5, ha="right")
    a3.set_xlabel("Model score of raised alerts"); a3.set_ylabel("Alerts")
    a3.set_title("(c) Confidence of alerts, all unseen days", fontsize=11)
    fig.suptitle("Production monitoring baselines measured on 2.5 million unseen flows",
                 fontweight="bold", fontsize=13)
    save_fig(fig, "fig17_monitoring")

    save_json({"model": best, "threshold": thr,
               "psi_bands": {"watch": PSI_WATCH, "investigate": PSI_INVESTIGATE},
               "features_investigate": int((worst.max_psi_across_days > PSI_INVESTIGATE).sum()),
               "features_watch": int(((worst.max_psi_across_days > PSI_WATCH) &
                                      (worst.max_psi_across_days <= PSI_INVESTIGATE)).sum()),
               "alert_volume": day_rows,
               "median_alerts_per_day": med_alerts, "max_alerts_per_day": max_alerts,
               "quiet_days": quiet_days, "n_days": int(len(vol))}, "monitoring")
    print("Saved drift, alert-volume, monitoring-plan and response-workflow tables.")


if __name__ == "__main__":
    main()
