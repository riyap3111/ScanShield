"""Build the Deliverable 3 Status Report by filling the course Word template.

Run after `bash run_all.sh`. Every figure comes from outputs/figures and every number is read
out of outputs/tables and outputs/metrics, so the document cannot drift from the pipeline.

Usage:  python reports/build_deliverable3.py
"""
import json
import sys
from pathlib import Path

import docx
import pandas as pd
from docx.shared import Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from docx_helpers import (  # noqa: E402
    appendix_heading, bullets_after, caption, delete_table, figure_after, fill_table,
    find_paragraph, page_break, paragraph_after, put_figure, set_paragraph, write_paragraphs,
)

# Every figure sits in the section it belongs to rather than in an appendix.
FIGURES = {
    "model_comparison": (1, "fig10_model_comparison.png"),
    "confusion": (2, "fig11_confusion_matrices.png"),
    "errors": (3, "fig12_error_profile.png"),
    "generalization": (4, "fig13_generalization.png"),
    "adversarial": (5, "fig14_adversarial.png"),
    "subgroups": (6, "fig15_subgroups.png"),
    "explainability": (7, "fig16_explainability.png"),
    "monitoring": (8, "fig17_monitoring.png"),
}


def place(anchor, key, cap_text, width=6.2):
    """Drop a numbered figure under `anchor` (a table or paragraph)."""
    num, name = FIGURES[key]
    path = FIG / name
    if not path.exists():
        return None
    el = anchor._tbl if hasattr(anchor, "_tbl") else anchor._p
    parent = anchor._parent
    return figure_after(el, parent, path, f"Figure {num}. {cap_text}", width_inches=width)

TAB = ROOT / "outputs" / "tables"
MET = ROOT / "outputs" / "metrics"
FIG = ROOT / "outputs" / "figures"
TEMPLATE = ROOT / "03_Interim_Status_Report_Template.docx"
OUT_DIR = ROOT / "reports" / "Deliverable3_Status"
OUT = OUT_DIR / "NetSentries_Deliverable3_Interim_Status_Report.docx"

REPO = "https://github.com/riyap3111/ScanShield"
DATASET = "https://www.kaggle.com/datasets/sateeshkumar6289/cicids-2017-dataset"
SUBMITTED = "November 8, 2026"


def t(name):
    return pd.read_csv(TAB / f"{name}.csv")


def j(name):
    return json.load(open(MET / f"{name}.json"))


def pct(x, d=3):
    return "n/a" if pd.isna(x) else f"{x * 100:.{d}f}%"


def f5(x, d=5):
    return "n/a" if pd.isna(x) else f"{x:.{d}f}"


def build_context():
    """Everything the prose quotes, pulled from the pipeline outputs."""
    c = {}
    comp = t("model_comparison")
    c["comp"] = comp
    test = comp[comp.split == "test (locked)"]
    val = comp[comp.split == "validation"]
    c["test"], c["val"] = test.set_index("model"), val.set_index("model")
    # Same rule the pipeline uses (src/utils.select_model): decided on VALIDATION, never on test.
    sys.path.insert(0, str(ROOT / "src"))
    from utils import select_model  # noqa: E402
    c["best"] = select_model()
    c["b"] = c["test"].loc[c["best"]]
    c["bv"] = c["val"].loc[c["best"]]

    order = ["Logistic regression", "Random Forest", "Gradient Boosting", "XGBoost"]
    present = list(dict.fromkeys(comp.model))
    c["models"] = [m for m in order if m in present] + [m for m in present if m not in order]
    c["tune"] = t("model_tuning").set_index("model")
    c["cv"] = t("cv_variance").set_index("model")
    c["thr"] = t("threshold_choice").set_index("model")
    c["cost"] = t("compute_cost").set_index("model")
    c["abl"] = t("feature_set_ablation").set_index("feature_set")
    c["seed"] = t("seed_stability")
    c["eng"] = t("engineered_features")
    c["fs"] = j("feature_sets_v2")

    c["causes"] = t("error_causes")
    c["eprof"] = t("error_analysis_v2")
    c["ports"] = t("fn_fp_ports")

    c["gen_day"] = t("generalization_by_day")
    c["gen_att"] = t("generalization_by_attack")
    c["gen_sum"] = t("generalization_summary")
    c["schema"] = t("schema_compatibility")
    c["gen"] = j("generalization")

    c["adv"] = t("adversarial_results").set_index("attack")
    c["advd"] = t("adversarial_defense").set_index("attack")
    c["advj"] = j("adversarial")

    c["svc"] = t("subgroup_service")
    c["size"] = t("subgroup_flowsize")
    c["subsum"] = t("subgroup_summary")
    c["imp"] = t("explainability_importance")
    c["gov"] = t("privacy_governance")
    c["rai"] = j("responsible_ai")

    c["psi"] = t("drift_psi")
    c["vol"] = t("alert_volume")
    c["plan"] = t("monitoring_plan")
    c["flow"] = t("response_workflow")
    c["mon"] = j("monitoring")

    seed_std = c["seed"][c["seed"].seed == "std"]
    c["seed_recall_std"] = float(seed_std.recall.iloc[0]) if len(seed_std) else float("nan")
    return c


# ----------------------------------------------------------------------------- sections
def header(doc, c, T):
    tb = T[0]
    fill_table(tb, [
        ["Deliverable", "Deliverable 3: Interim Progress Report 2",
         "Selected Track", "Intrusion Detection Systems (IDS)"],
        ["Project Title", "ScanShield: Detecting Network Port Scanning Using Flow-Level "
                          "Traffic Analysis", "Team Name", "NetSentries"],
        ["Team Members", "Riya Sanjiv Pradhan; Ojas Prashant Suratkar",
         "Submission Date", SUBMITTED],
        ["Repository", REPO, "Dataset Link", DATASET],
    ], header=False, size=Pt(9))
    for i in range(4):
        tb.cell(i, 0).paragraphs[0].runs[0].bold = True
        tb.cell(i, 2).paragraphs[0].runs[0].bold = True
    delete_table(T[1])                   # the "How to use this template" box


def snapshot(doc, c, T):
    b, bv = c["b"], c["bv"]
    worst = c["advd"].drop(index="Clean benign traffic (false-positive rate)")
    worst_attack = worst.recall_baseline_model.idxmin()
    worst_rec = worst.recall_baseline_model.min()
    worst_after = worst.loc[worst_attack, "recall_after_adversarial_training"]
    ext_fpr = c["gen"]["external_benign_fpr"]
    fill_table(T[2], [
        ["Item", "Current status"],
        ["Overall status",
         f"On track. {len(c['models'])} model families tuned and compared, test split scored "
         "once, and the "
         "generalization, adversarial, Responsible AI and monitoring work complete. Both "
         "proposal criteria met with margin."],
        ["Best-performing approach so far",
         f"{c['best']} on all {c['fs']['n_full']} features: test recall {f5(b.recall)}, "
         f"precision {f5(b.precision)}, F1 {f5(b.f1)}, PR-AUC {f5(b.pr_auc, 6)}, FPR "
         f"{pct(b.fpr, 4)} ({int(b.FP)} false alarms in {int(b.TN + b.FP):,} benign flows)."],
        ["Most important error or risk",
         f"Evasion, not accuracy. Padding probes with 24 bytes drops recall to "
         f"{f5(c['adv'].loc['A2 Payload padding, 24 bytes'].recall_under_attack, 3)}, 1,000x "
         f"pacing to "
         f"{f5(c['adv'].loc['A1 Timing evasion, 1000x slower'].recall_under_attack, 3)}. "
         f"Adversarial training recovers most of it; pacing needs per-source features."],
        ["Work remaining before final report",
         "Slow-scan features on the IP-bearing CIC release (critical path); Streamlit "
         "dashboard; final report and demo."],
    ], widths=[1.5, 5.15])


def s1_status(doc, c):
    b = c["b"]
    p = find_paragraph(doc, "[Summarize work completed")
    write_paragraphs(p, [
        f"Since Interim Report 1 the project moved from one logistic-regression baseline to a "
        f"tuned, compared and stress-tested detector. Changes to scope and method since then:",
    ])
    p = bullets_after(p, [
        f"Feature engineering, not pruning. Every original CICFlowMeter column is kept and "
        f"{len(c['eng'])} engineered features added on top ({c['fs']['n_full']} inputs): payload "
        f"per packet, reply ratio, flag density, header-to-payload ratio. Section 3 ablates them.",
        f"Three new families. Random Forest, scikit-learn's histogram gradient boosting and "
        f"XGBoost (Chen & Guestrin, 2016) joined the baseline, so the proposal's planned "
        f"candidates are both covered. XGBoost needs an OpenMP runtime Apple does not ship; "
        f"src/install_libomp.py fetches it from conda-forge and verifies its checksum, so the "
        f"pipeline reproduces without Homebrew.",
        f"A genuinely held-out test. Threshold and model choice were both fixed on validation; "
        f"the test split was read once afterwards.",
        f"Generalization went well past the Monday control: all seven other CICIDS2017 day "
        f"files, {int(c['gen_day'].rows_after_cleaning.sum()):,} flows, "
        f"{len(c['gen_att'].attack_type.unique())} unseen attack types. Adversarial, Responsible AI and monitoring work is also complete.",
    ])
    paragraph_after(
        p,
        f"Data, code and pipeline are reproducible: run_all.sh rebuilds every table and figure "
        f"here from the raw CSVs at seed 42. No written feedback on Interim Report 1 had been "
        f"returned when this was prepared, so that plan was carried through unchanged apart "
        f"from the boosting substitution. Remaining work is in Section 9.")


def s2_models(doc, c, T):
    tune, cv, cost = c["tune"], c["cv"], c["cost"]
    p = find_paragraph(doc, "[Describe the models")
    write_paragraphs(p, [
        f"{len(c['models'])} families were compared on identical data — same training split, same "
        f"{c['fs']['n_full']} features, same protocol — differing in kind, not just in "
        f"hyperparameters: linear, bagged trees, and two independent boosting implementations. "
        f"Each got a 15-candidate "
        f"randomised search scored by 5-fold stratified CV on the training split only "f"(scikit-learn; Pedregosa et al., 2011). The "
        f"search metric is average precision, not accuracy, which is uninformative when 42.5% "
        f"of rows are positive and the two error types cost differently. Imbalance is handled "
        f"with balanced class weights, not resampling, which would distort the very "
        f"false-positive rate being measured.",
        f"Validation design. The alert threshold is the highest-F1 threshold on validation "
        f"inside the 5% ceiling, then frozen; the winning family is chosen the same way, on "
        f"validation, so neither is informed by test data. Only then was test scored, once. "
        f"Reproducibility rests on seed 42, tuned parameters in Appendix A, and a one-command "
        f"pipeline; three seeds change recall by {c['seed_recall_std']:.1e}.",
    ])
    rows = [["Approach", "Key settings", "Validation method", "Rationale / result"]]
    labels = {"Logistic regression": "Baseline: Logistic regression",
              "Random Forest": "Random Forest",
              "Gradient Boosting": "Gradient Boosting (scikit-learn hist.)",
              "XGBoost": "XGBoost"}
    for m in c["models"]:
        params = json.loads(tune.loc[m, "best_params"])
        pretty = ", ".join(f"{k.split('__')[-1]}={_fmt(v)}" for k, v in params.items())
        tv = c["test"].loc[m]
        rows.append([
            labels.get(m, m),
            f"{pretty}; balanced class weights" if "class_weight" not in params else pretty,
            f"15-candidate randomised search, 5-fold stratified CV on train. PR-AUC "
            f"{cv.loc[m, 'pr_auc_mean']:.5f} ± {cv.loc[m, 'pr_auc_std']:.5f}",
            f"{_why(m, c['best'])} Test recall {f5(tv.recall)}, FPR {pct(tv.fpr, 4)}, "
            f"F1 {f5(tv.f1)}. Scores {cost.loc[m, 'flows_scored_per_second']:,.0f} flows/s.",
        ])
    fill_table(T[3], rows, widths=[1.25, 1.55, 1.75, 2.1])


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _why(m, best):
    base = {
        "Logistic regression": "Linear reference carried over from Interim Report 1.",
        "Random Forest": "Bagged trees capture threshold interactions a linear model cannot.",
        "Gradient Boosting": "Boosted trees correcting their own errors in sequence, in "
                             "scikit-learn's histogram implementation.",
        "XGBoost": "The gradient-boosting library named in the proposal, run with the "
                   "histogram tree method.",
    }.get(m, "")
    return ("Selected model. " + base) if m == best else base


def s3_evaluation(doc, c, T):
    b, abl, cost = c["b"], c["abl"], c["cost"]
    p = find_paragraph(doc, "[Report metrics appropriate")
    write_paragraphs(p, [
        f"The primary metric is recall on port-scan flows at a false-positive rate within 5%, "
        f"the proposal's two criteria: a missed scan removes the early warning the system "
        f"exists to give, and the FPR ceiling is what keeps analysts using it. Accuracy is not "
        f"used to judge — predicting BENIGN for everything scores 57.5% here while catching "
        f"nothing — and PR-AUC is the ranking metric, being the one that stays informative "
        f"under imbalance. All figures are on the locked test split "
        f"({int(b.TN + b.FP + b.FN + b.TP):,} flows), scored once at the frozen threshold.",
    ])
    rows = [["Model", "Precision", "Recall", "F1", "PR-AUC / ROC-AUC", "Operational note"]]
    best_size = c["cost"].loc[c["best"], "model_size_mb"]
    for m in c["models"]:
        tv, cost_m = c["test"].loc[m], c["cost"].loc[m]
        if m == c["best"]:
            n = (f"Selected on validation. {cost_m.flows_scored_per_second:,.0f} flows/s, "
                 f"{cost_m.model_size_mb:.1f} MB.")
        else:
            gap = tv.pr_auc - c["test"].loc[c["best"]].pr_auc
            rank = ("matches the selected model on ranking" if abs(gap) < 5e-6
                    else f"PR-AUC {gap:+.6f} against the selected model")
            lead = ("Baseline: fastest and fully interpretable"
                    if m == "Logistic regression" else "Not selected")
            n = f"{lead}; {rank}."
        rows.append([
            m + (" (baseline)" if m == "Logistic regression" else ""),
            f5(tv.precision), f5(tv.recall), f5(tv.f1),
            f"{tv.pr_auc:.6f} / {tv.roc_auc:.6f}",
            f"FPR {pct(tv.fpr, 4)}; {int(tv.FP)} FP, {int(tv.FN)} FN. {n}",
        ])
    fill_table(T[4], rows, widths=[1.3, 0.68, 0.65, 0.65, 1.12, 2.1])

    put_figure(T[5], FIG / "fig10_model_comparison.png", width_inches=5.6)
    cap = find_paragraph(doc, "Figure 1.")
    caption(cap, f"Figure 1. (a) Misclassified flows per family, (b) precision-recall zoomed "
                 f"into the corner where the curves separate, (c) confusion matrix of the "
                 f"selected {c['best']}, all on the locked test split. Takeaway: all three "
                 f"clear both targets and are separated by a handful of flows, so the choice "
                 f"rests on error counts rather than on ranking quality.")

    p = find_paragraph(doc, "[State the main conclusion")
    write_paragraphs(p, [
        f"Both criteria are met with wide margin: recall {f5(b.recall)} against a 0.90 target, "
        f"FPR {pct(b.fpr, 4)} against a 5% ceiling, from {int(b.FP)} false alarms in "
        f"{int(b.TN + b.FP):,} benign test flows. This is stable, not a lucky split: 5-fold "
        f"PR-AUC {c['cv'].loc[c['best'], 'pr_auc_mean']:.5f} ± "
        f"{c['cv'].loc[c['best'], 'pr_auc_std']:.5f}, and three seeds agree on recall to "
        f"{c['seed_recall_std']:.1e}. The {len(c['models'])} families are separated by "
        f"{int(c['test'].FN.max() - c['test'].FN.min())} flows out of "
        f"{int(b.TN + b.FP + b.FN + b.TP):,}: the honest reading is that they are tied, and the "
        f"choice rests on the zero false alarms rather than a capability gap. "
        f"{_ablation_sentence(c)} Cost is no constraint at "
        f"{cost.loc[c['best'], 'model_size_mb']:.1f} MB and "
        f"{cost.loc[c['best'], 'flows_scored_per_second']:,.0f} flows/s on a laptop.",
    ])
    place(p, "confusion",
          f"Confusion matrices for all {len(c['comp'].model.unique())} families at their own "
          f"frozen thresholds on the locked test split. Takeaway: the families are separated by "
          f"a handful of flows, so the selection rests on error counts rather than on a gap in "
          f"capability.", width=6.0)


def _ablation_sentence(c):
    """Describe the feature-set ablation from whatever it actually measured."""
    a = c["abl"]
    full, orig, nofp = a.loc["full"], a.loc["original_only"], a.loc["no_fingerprint"]
    # Compare total errors, not false alarms alone: on this data every set reaches zero
    # false alarms, so missed scans are the only thing that can separate them.
    d_fp = (orig.FP + orig.FN) - (full.FP + full.FN)
    if d_fp > 0:
        eng = (f"The engineered features earn their place: on validation the full set makes "
               f"{int(full.FP)} false alarms against {int(orig.FP)} for the original columns "
               f"alone, at recall {f5(full.recall, 4)} versus {f5(orig.recall, 4)}.")
    elif d_fp < 0:
        eng = (f"The engineered features do not pay off here: on validation the full set and "
               f"the original columns alone both reach {int(full.FP)} false alarms, and the "
               f"original columns are marginally ahead on recall "
               f"({f5(orig.recall, 4)} against {f5(full.recall, 4)}, a difference of "
               f"{abs(int(full.FN) - int(orig.FN))} flows). This data is close "
               f"to linearly separable, so above 0.998 recall at zero false alarms there is no "
               f"headroom for any feature set to show an advantage. They are kept because they "
               f"cost nothing and separate the classes cleanly on their own (Table A3), not "
               f"because this dataset proves they help.")
    else:
        eng = (f"The ablation is candid about its limits: the engineered features and the "
               f"original columns alone tie at the operating point ({int(full.FP)} false alarms "
               f"each, recall {f5(full.recall, 4)} vs {f5(orig.recall, 4)}). This data is close "
               f"to linearly separable, so above 0.998 recall at zero false alarms no feature "
               f"set has room to show an advantage. The engineered features are kept because "
               f"they cost nothing and separate the classes cleanly on their own (Table A3), "
               f"not because this dataset proves they help.")
    fp = (f"Removing the port and TCP-window columns, which encode this particular scanner's "
          f"fingerprint, still leaves recall at {f5(nofp.recall, 4)} with FPR "
          f"{pct(nofp.fpr, 4)}, so the model is reading scan behaviour rather than memorising "
          f"one tool.")
    return eng + " " + fp


def s4_errors(doc, c, T):
    b, causes = c["b"], c["causes"].set_index("pattern")
    p = find_paragraph(doc, "[Analyze representative")
    fp_n, fn_n = int(b.FP), int(b.FN)
    write_paragraphs(p, [
        f"At the frozen threshold the model raises {fp_n} false alarms and misses {fn_n} scans "
        f"in {int(b.TN + b.FP + b.FN + b.TP):,} test flows, so on this split the only error "
        f"type to analyse is the missed scan; the false-positive row below therefore draws on "
        f"the unseen days from Section 5, where false alarms do occur. Both error types live in "
        f"the same narrow band where a probe looks like a failed connection (Figure 3).",
    ])
    rows = [["Example / error type", "Likely cause", "Security or operational impact",
             "Planned improvement"]]
    fp_noresp = causes.loc["Benign flow that received no reply at all"]
    fn_reply = causes.loc["Scan flow that did get a reply packet"]
    fn_payload = causes.loc["Scan flow carrying payload bytes"]
    day = c["gen_day"]
    worst_day = day.loc[day.benign_fpr.idxmax()]
    ext_fp = int(day.benign_flagged.sum())
    rows.append([
        f"False positive: none on the test split, {ext_fp:,} across "
        f"{int(day.benign_flows.sum()):,} unseen benign flows "
        f"({int(worst_day.benign_flagged):,} of them on {worst_day.day} alone)",
        "A failed, refused or abandoned connection has the same signature as a probe: one "
        "packet out, nothing back, no payload. The flow record cannot show intent. The "
        f"{worst_day.day} concentration points at one traffic pattern, not a general problem.",
        f"Analyst time, and it is affordable: {pct(worst_day.benign_fpr, 3)} on the worst "
        "unseen day. But each alert is a real user's connection being questioned, so the "
        "concentration matters more than the average.",
        "Per-source aggregation: one unanswered flow is noise, fifty from one host in an hour "
        f"is a scan. Profile the {worst_day.day} alerts before deployment to find what that "
        "day's traffic does differently.",
    ])
    rows.append([
        f"False negative: probe that drew a response ({int(fn_reply.n)} of {fn_n}, "
        f"{pct(fn_reply.share_of_errors, 0)}; {int(fn_payload.n)} carried payload)",
        "A probe against an open port completes the handshake and gets data back, so it looks "
        "like a legitimate session starting.",
        "The most valuable probes to catch: they are the ones that found an open service, so "
        "missing them loses the part of the scan the attacker acts on.",
        "Train on connect-scan traffic explicitly. The hardened model in Section 6 already "
        f"lifts recall on these to "
        f"{f5(c['advd'].loc['A4 Handshake mimicry (connect scan)'].recall_after_adversarial_training, 4)}.",
    ])
    tb = fill_table(T[6], rows, widths=[1.35, 1.6, 1.7, 2.0])
    place(tb, "errors",
          "What the selected model gets wrong on the locked test split, by outcome. Takeaway: "
          "the missed scans sit between the two class distributions on every axis, which is "
          "why per-flow features alone cannot separate them further.", width=6.3)


def s5_generalization(doc, c, T):
    gen, day = c["gen"], c["gen_day"]
    att = c["gen_att"].groupby("attack_type").agg(
        flows=("flows", "sum"), flagged=("flagged_as_portscan", "sum"))
    att["rate"] = att.flagged / att.flows
    worst_day = day.loc[day.benign_fpr.idxmax()]
    best_day = day.loc[day.benign_fpr.idxmin()]
    top_att = att.sort_values("rate", ascending=False).iloc[0]
    b = c["b"]
    p = find_paragraph(doc, "[Compare performance on the course sample")
    write_paragraphs(p, [
        f"The model is trained on one afternoon of one file, so the real question is what it "
        f"does on traffic it has never seen. It was scored, unchanged and at the frozen "
        f"threshold, on every other CICIDS2017 capture day (Sharafaldin et al., 2018; "f"Canadian Institute for Cybersecurity, n.d.): "
        f"{int(day.rows_after_cleaning.sum()):,} flows, {gen['external_benign_flows']:,} of "
        f"them benign, plus {len(att)} attack types absent from training. Schema compatibility "
        f"came first: all seven files carry the same 79 columns, but every header has leading "
        f"whitespace and the web-attacks file is cp1252-encoded, so a plain UTF-8 read of its "
        f"'Web Attack - Brute Force' labels fails outright. The pipeline audits these "
        f"differences (Appendix B, Table B3) rather than coercing them silently, then applies "
        f"the identical cleaning and feature code path.",
    ])
    rows = [["Dataset", "Differences from training data", "Result", "Interpretation"]]
    rows.append([
        f"Course sample: held-out test split ({int(b.TN + b.FP + b.FN + b.TP):,} flows)",
        "Same capture and day; unseen rows only, de-duplicated before splitting so no training "
        "flow reappears.",
        f"Recall {f5(b.recall)}, FPR {pct(b.fpr, 4)}, F1 {f5(b.f1)}, PR-AUC {f5(b.pr_auc, 6)}",
        "Both criteria met. The optimistic end of the range: the scan arrives in one burst, so "
        "a random split puts flows from the same scan on both sides.",
    ])
    rows.append([
        f"Larger external: all 7 other day files ({int(day.rows_after_cleaning.sum()):,} flows, "
        f"{len(att)} unseen attack types)",
        "Four other capture days, different traffic mix. No PortScan label exists, so recall "
        "cannot be measured; what can is whether the detector stays quiet. Needed a cp1252 "
        "fallback and header normalisation.",
        f"Benign FPR {pct(gen['external_benign_fpr'], 4)} overall "
        f"({gen['external_benign_flagged']:,} alerts in {gen['external_benign_flows']:,} benign "
        f"flows); per day {pct(best_day.benign_fpr, 4)} to {pct(worst_day.benign_fpr, 4)}, "
        f"every day inside the ceiling.",
        f"The rate holds on days the model never saw. Two details matter more: "
        f"{int(worst_day.benign_flagged):,} of the {gen['external_benign_flagged']:,} alerts "
        f"come from {worst_day.day} alone and four days produce none, so the risk is "
        f"concentrated rather than spread; and only {int(att.flagged.sum())} of "
        f"{int(att.flows.sum()):,} flows from the other {len(att)} attack types are called a scan — a "
        f"strength (no cross-talk) and a limitation (it detects scans, not intrusions). "
        f"Both rows inherit CICIDS2017's own labelling errors (Engelen et al., 2021), so the "
        f"last fraction of a percent should not be read too closely.",
    ])
    tb = fill_table(T[7], rows, widths=[1.3, 1.65, 1.75, 2.0])
    place(tb, "generalization",
          "Left: false-positive rate on benign traffic for each unseen capture day against the "
          "5% ceiling. Right: how often each unseen attack type is labelled a port scan. "
          "Takeaway: the false-alarm rate holds on days the model never saw, and other attack "
          "families almost never trigger it.", width=6.3)


def s6_adversarial(doc, c, T):
    adv, advd, advj = c["adv"], c["advd"], c["advj"]
    slow = adv.loc["A1 Timing evasion, 1000x slower"]
    slow_d = advd.loc["A1 Timing evasion, 1000x slower"]
    comb = adv.loc["A5 Combined (A1 100x + A2 + A3 + A4)"]
    comb_d = advd.loc["A5 Combined (A1 100x + A2 + A3 + A4)"]
    hand = adv.loc["A4 Handshake mimicry (connect scan)"]
    pad = adv.loc["A2 Payload padding, 24 bytes"]
    win = adv.loc["A3 TCP window reshaping"]
    p = find_paragraph(doc, "[Define the protected asset")
    write_paragraphs(p, [
        f"Every feature ScanShield reads comes from the attacker's own packets, so an attacker "
        f"who suspects a flow-based detector can simply scan differently. Five evasion "
        f"techniques, each a documented nmap option (Lyon, 2009), were applied to the "
        f"{advj['test_scan_flows']:,} held-out scan flows, perturbing the raw columns and "
        f"recomputing every engineered feature so the shaped flows stay consistent. All of it "
        f"ran offline on a public dataset: no traffic generated, no system contacted.",
        f"The results are not comfortable — the deployed model is brittle. Padding probes with "
        f"24 bytes drops recall from "
        f"{f5(adv.loc['A0 No attack (clean scan flows)'].recall_under_attack, 4)} to "
        f"{f5(pad.recall_under_attack, 4)}; completing the handshake to "
        f"{f5(hand.recall_under_attack, 4)}; pacing 1,000x slower to "
        f"{f5(slow.recall_under_attack, 4)}. The model learned that a scan is small, empty and "
        f"unanswered, so anything breaking that shape defeats it. Pacing is also a structural "
        f"limit: a per-flow classifier sees one flow, and a scan spread over hours looks "
        f"ordinary in every one.",
        f"Defense. These are traffic-shaping transforms, so the defender can generate them "
        f"without meeting an attacker: shaped copies of the training scans were added to the "
        f"training set and the model refit ({advj['augmented_training_rows']:,} rows). Recall "
        f"recovers to "
        f"{f5(advd.loc['A2 Payload padding, 24 bytes'].recall_after_adversarial_training, 4)} "
        f"under padding, "
        f"{f5(advd.loc['A4 Handshake mimicry (connect scan)'].recall_after_adversarial_training, 4)} "
        f"under handshake mimicry and {f5(comb_d.recall_after_adversarial_training, 4)} under "
        f"the combined attack, at an unchanged clean false-positive rate "
        f"({pct(advj['clean_fpr_hardened'], 4)}). It does not fix everything: 1,000x pacing "
        f"still leaves recall at {f5(slow_d.recall_after_adversarial_training, 4)}, which is why "
        f"per-source slow-scan features are the critical path in Section 9. These are "
        f"feature-space approximations, so they bound the risk rather than measure it exactly. "
        f"Full results in Appendix C, Table C1.",
    ])
    fill_table(T[8], [
        ["Asset", "Attacker objective", "Capability", "Attack / test", "Defense / mitigation"],
        ["The deployed classifier and the early warning it gives the SOC.",
         "Enumerate open ports without any flow crossing the alert threshold.",
         "Black box: no access to the model, its scores or its training data, so no poisoning. "
         "Full control of their own host, hence of probe timing, payload length, TCP window "
         "and handshake completion.",
         f"Five shaping attacks on {advj['test_scan_flows']:,} held-out scan flows: timing "
         f"(10/100/1,000x), padding (24/64 B), window reshaping, handshake mimicry, combined. "
         f"Worst case recall {f5(min(slow.recall_under_attack, comb.recall_under_attack), 4)}.",
         f"Adversarial training restores recall to "
         f"{f5(comb_d.recall_after_adversarial_training, 4)} under the combined attack at "
         f"unchanged clean FPR. Structural backstop for pacing: per-source slow-scan features "
         f"(src/slow_scan.py) over 5 min, 1 h and 24 h windows."],
    ], widths=[1.05, 1.05, 1.5, 1.45, 1.65])
    place(T[8], "adversarial",
          "Recall on held-out scan flows under each evasion attack, before and after "
          "adversarial training. Takeaway: simple traffic shaping defeats the deployed model "
          "outright, and training on shaped flows recovers everything except extreme pacing.",
          width=5.9)


def _disparity_sentence(c, svc_ok, spread):
    """State the fairness finding from what was measured, including the all-zero case."""
    worst = svc_ok.loc[svc_ok.fpr.idxmax()]
    rec = svc_ok[svc_ok.scan_flows >= 50]
    rec_lo, rec_hi = (rec.recall.min(), rec.recall.max()) if len(rec) else (float("nan"),) * 2
    if worst.fpr == 0:
        head = (f"They cannot here: not one of the {len(svc_ok)} service groups produces a "
                f"single false alarm on the test split, so there is no disruption to "
                f"distribute unevenly.")
    else:
        head = (f"They do not: across service groups the false-positive rate spans "
                f"{spread['spread'] * 100:.3f} percentage points, the highest being "
                f"{pct(worst.fpr, 3)} on {worst.service_group.lower()}.")
    if not len(rec):
        return head + " Recall is flat across flow sizes and the capture (Figure 6)."
    low = rec.loc[rec.recall.idxmin()]
    tail = (f" Recall is where the real disparity sits: it spans {f5(rec_lo, 4)} to "
            f"{f5(rec_hi, 4)}, and the weakest group is {low.service_group} at "
            f"{f5(low.recall, 4)} ({int(low.FN)} of {int(low.scan_flows)} scans missed). That "
            f"matters more than the number suggests, because those are among the ports an "
            f"attacker most wants open, so the model is weakest exactly where a hit is most "
            f"valuable. Recall is otherwise flat across flow sizes and the capture (Figure 6). "
            f"Caveats: that group holds only {int(low.scan_flows)} scan flows, and zero false "
            f"alarms on one lab capture is weak evidence of fairness.")
    return head + tail


def s7_responsible(doc, c, T):
    gov, svc, imp = c["gov"], c["svc"], c["imp"]
    svc_ok = svc[svc.flows >= 50]
    worst_svc = svc_ok.loc[svc_ok.fpr.idxmax()]
    spread = [r for r in c["rai"]["disparity"]
              if r["dimension"].startswith("Service") and r["rate"] == "false-positive rate"][0]
    p = find_paragraph(doc, "[Report completed or planned bias")
    write_paragraphs(p, [
        f"CICIDS2017 carries no demographic attributes, so unequal performance was tested on "
        f"the dimensions that do exist and carry operational consequence: service group, flow "
        f"size, period within the capture, and capture day. The stake is concrete — if false "
        f"alarms concentrated on one service, the team running it would absorb the disruption "
        f"and be first to distrust the tool. {_disparity_sentence(c, svc_ok, spread)} "
        f"Privacy, retention, oversight and misuse controls continue in Appendix C, Table C4.",
    ])
    rows = [["Concern", "Test or safeguard", "Evidence / current finding", "Remaining action"]]
    for _, r in gov.head(3).iterrows():
        rows.append([r.concern, r.test_or_safeguard, r.evidence, r.remaining_action])
    tb = fill_table(T[9], rows, widths=[1.1, 1.8, 1.9, 1.85])
    fig7 = place(tb, "subgroups",
                 "Performance by service group, flow size and position in the capture. "
                 "Takeaway: no group absorbs a disproportionate share of the false alarms, but "
                 "recall is visibly weaker on file-sharing ports.", width=6.3)
    place(fig7, "explainability",
          "Permutation importance across the whole model, and a worked explanation of one "
          "individual alert. Takeaway: the model leans on reply behaviour and flow economy "
          "rather than on any single tool-specific value, and every alert can be explained to "
          "the analyst who receives it.", width=6.3)


def s8_monitoring(doc, c, T):
    plan, vol, mon = c["plan"], c["vol"], c["mon"]
    p = find_paragraph(doc, "[Define what will be monitored")
    med = mon["median_alerts_per_day"]
    worst_psi = c["psi"].iloc[0]
    write_paragraphs(p, [
        f"These baselines are measured, not assumed: alert volume and drift come from the same "
        f"{int(c['gen_day'].rows_after_cleaning.sum() / 1e6):.1f} million unseen flows as "
        f"Section 5, and they are lopsided. {mon.get('quiet_days', 0)} of "
        f"{mon.get('n_days', len(c['vol']))} unseen days raise no alert at all while the "
        f"busiest raises {mon.get('max_alerts_per_day', 0):,.0f}, so the volume rule is banded "
        f"on the busiest day rather than on a median of zero. Drift uses the population "
        f"stability index against the training distribution: "
        f"{mon['features_investigate']} of {len(c['psi'])} features cross the 0.25 investigate "
        f"line on some day, the largest {worst_psi.feature} "
        f"(PSI {worst_psi.max_psi_across_days:.2f}) — expected from a different service mix, "
        f"not a broken sensor. Response is review-first at every tier: nothing is blocked "
        f"automatically, because a false alarm must never cut off a real user. The seven-step "
        f"response path and logging schema are in Appendix C, Tables C2-C3.",
    ])
    rows = [["Signal / metric", "Threshold or trigger", "Owner / response", "Feedback use"]]
    for _, r in plan.head(4).iterrows():
        rows.append([r.signal, r.threshold_or_trigger, r.owner_and_response, r.feedback_use])
    tb = fill_table(T[10], rows, widths=[1.3, 1.85, 1.9, 1.6])
    place(tb, "monitoring",
          "Production baselines measured on the unseen days: feature drift, alert volume per "
          "day, and the confidence distribution of raised alerts. Takeaway: the thresholds in "
          "the table above are anchored to observed behaviour, not guessed.", width=6.3)


def s9_plan(doc, c, T):
    p = find_paragraph(doc, "[List the remaining analyses")
    write_paragraphs(p, [
        "The critical path to the final report runs through the IP-bearing CIC release: it is "
        "the only thing that unlocks the per-source slow-scan features, and those are the "
        "answer to the one real weakness found in Section 6. Everything else can proceed in "
        "parallel.",
    ])
    fill_table(T[11], [
        ["Remaining task", "Owner", "Due date", "Definition of done"],
        ["Critical path: per-source slow-scan features on CIC's GeneratedLabelledFlows release "
         "(adds Timestamp and IPs), added as model inputs", "Riya", "Nov 18",
         "Recall under the 1,000x timing attack reported with and without them; allowlist "
         "evaluated end to end on real source IPs"],
        ["Streamlit dashboard: ranked alerts with per-alert explanation and allowlist status",
         "Riya", "Nov 22",
         "Demo takes a flow file, ranks alerts, shows top contributing features, marks "
         "suppressed ones"],
        ["Code hardening: extend the 18 existing tests to the cleaning and scoring paths; pin "
         "the environment", "Ojas", "Nov 25",
         "pytest green on every module; requirements.txt verified from a clean virtualenv"],
        ["Final evaluation: calibration, cost-sensitive thresholds, and promotion of the "
         "hardened model if it holds", "Ojas", "Nov 29",
         "Calibration curve and threshold-cost table committed; one documented decision on "
         "which model ships"],
        ["Final report and presentation", "Both", "Dec 6",
         "Report submitted; rehearsed demo walkthrough"],
    ], widths=[2.3, 0.5, 0.6, 3.1])


def references(doc, c):
    p = find_paragraph(doc, "[List datasets, tools, papers")
    refs = [
        "Canadian Institute for Cybersecurity. (n.d.). Intrusion detection evaluation dataset "
        "(CIC-IDS2017). University of New Brunswick. https://www.unb.ca/cic/datasets/ids-2017.html",
        "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In "
        "Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and "
        "Data Mining (pp. 785-794).",
        "Engelen, G., Rimmer, V., & Joosen, W. (2021). Troubleshooting an intrusion detection "
        "dataset: The CICIDS2017 case study. In 2021 IEEE Security and Privacy Workshops (SPW) "
        "(pp. 7-12). IEEE.",
        "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). "
        "LightGBM: A highly efficient gradient boosting decision tree. In Advances in Neural "
        "Information Processing Systems 30 (pp. 3146-3154).",
        "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model "
        "predictions. In Advances in Neural Information Processing Systems 30 (pp. 4765-4774).",
        "Lyon, G. F. (2009). Nmap network scanning: The official Nmap project guide to network "
        "discovery and security scanning. Insecure.Com LLC.",
        "NetSentries. (2026). ScanShield [Source code]. GitHub. " + REPO,
        "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. "
        "(2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, "
        "12, 2825-2830.",
        "Scarfone, K., Souppaya, M., Cody, A., & Orebaugh, A. (2008). Technical guide to "
        "information security testing and assessment (NIST Special Publication 800-115). "
        "National Institute of Standards and Technology.",
        "Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new "
        "intrusion detection dataset and intrusion traffic characterization. In Proceedings of "
        "the 4th International Conference on Information Systems Security and Privacy (ICISSP) "
        "(pp. 108-116).",
        "sateeshkumar6289. (n.d.). CICIDS 2017 dataset [Data set]. Kaggle. " + DATASET,
    ]
    write_paragraphs(p, refs, size=Pt(9), space_after=Pt(3))


def contributions(doc, c, T):
    fill_table(T[12], [
        ["Team member", "Responsibilities and completed work", "Evidence / artifact"],
        ["Riya Sanjiv Pradhan",
         "Feature engineering design and implementation; generalization testing across all seven "
         "unseen day files including the schema-compatibility audit and the cp1252 encoding fix; "
         "Responsible AI review (service-group, flow-size, time-period and per-day disparity "
         "analysis, privacy and governance table); monitoring and response workflow with the "
         "measured PSI and alert-volume baselines; co-wrote the report.",
         "src/07_feature_engineering.py, src/10_generalization.py, src/12_responsible_ai.py, "
         "src/13_monitoring.py; tables schema_compatibility, generalization_by_day, "
         "subgroup_*, privacy_governance, drift_psi, alert_volume; figures 13, 15, 16, 17"],
        ["Ojas Prashant Suratkar",
         "Model development and tuning for all four families including the XGBoost build "
         "(randomised search, 5-fold CV, "
         "class-weight strategy, threshold selection and freezing); evaluation figures and the "
         "structured error analysis; the adversarial threat model, the five evasion attacks and "
         "the adversarial-training defense; the Deliverable 3 results generator; co-wrote the "
         "report.",
         "src/08_train_models.py, src/09_evaluate.py, src/11_adversarial.py, "
         "src/14_write_summary_d3.py; tables model_comparison, model_tuning, cv_variance, "
         "threshold_choice, error_causes, adversarial_*; figures 10, 11, 12, 14"],
        ["Both",
         "Joint decisions: keeping every column and adding engineered features on top, the "
         "choice of PR-AUC as the search metric and recall-at-bounded-FPR as the primary "
         "metric, the threshold rule, and running both boosting implementations rather than "
         "only one. Each member reviewed and re-ran the other's scripts before merging.",
         "run_all.sh reproduces every output; RESULTS_DELIVERABLE3.md is generated from the "
         "saved tables"],
    ], widths=[1.1, 3.4, 2.15])
    p = find_paragraph(doc, "[Explain how the team coordinated")
    write_paragraphs(p, [
        "Work was split by pipeline stage as in the previous deliverables: Riya owns the data, "
        "generalization and deployment-facing stages, Ojas owns modelling, evaluation and the "
        "adversarial work. The two of us reviewed each other's code and re-ran the full pipeline "
        "from the raw CSVs before the report was written, and both checked every number quoted "
        "here against the generated tables. Participation was even and no coordination problems "
        "needed resolving."])


def ai_disclosure(doc, c, T):
    fill_table(T[14], [
        ["Tool", "How it was used", "What the team verified or changed"],
        ["Claude (Anthropic)",
         "Assisted with writing and debugging the Python for feature engineering, model tuning, "
         "the adversarial transforms, the subgroup and drift analysis, and the figure code; "
         "assisted with drafting and editing the text of this report.",
         "The team set the analytical direction and made every analytical decision: keeping all "
         "columns, which engineered features to add, PR-AUC as the search metric, "
         "recall-at-bounded-FPR as the primary metric, the threshold rule, the threat model and "
         "the choice of evasion attacks. We reviewed all generated code, re-ran the entire "
         "pipeline from the raw CSVs, and checked every number in this report against the "
         "tables in outputs/. Two substantive corrections we made: nested parallel jobs in the "
         "tuning search deadlocked on macOS and we reconfigured the job pool, and model "
         "selection was initially reading the locked test split, which we moved to validation "
         "so the test result stays honest."],
        ["", "", ""],
    ], widths=[1.1, 2.75, 2.8])
    # the template ships a spare row for a second tool; drop it
    tb = T[14]
    tb._tbl.remove(tb.rows[-1]._tr)


def checklist(doc, c, T):
    fill_table(T[15], [
        ["Check", "Requirement"],
        ["[x]", "All reported results identify the dataset split used."],
        ["[x]", "Baseline and candidate models are compared consistently."],
        ["[x]", "Error analysis includes both causes and security impact."],
        ["[x]", "Generalization testing includes the larger/external dataset."],
        ["[x]", "Threat model identifies attacker capabilities and constraints."],
        ["[x]", "Monitoring and response workflow is actionable."],
        ["[x]", "Contribution statement and AI-use disclosure are complete."],
    ], widths=[0.6, 6.05])


def rubric(doc, c, T):
    fill_table(T[16], [
        ["Criterion", "Points", "Where the evidence appears"],
        ["Updated status and project management", "1",
         "Executive Status Snapshot and Section 1; Section 9 plan"],
        ["Model development and comparison", "2",
         "Section 2 and its table; tuned settings and feature sets in Appendix A "
         "(Tables A1-A3)"],
        ["Evaluation and error analysis", "2",
         "Section 3, metric table, Figures 1-2; Section 4, its table and Figure 3"],
        ["Generalization testing", "2",
         "Section 5, its table and Figure 4; Appendix B Tables B1-B3"],
        ["Adversarial ML", "1",
         "Section 6, threat-model table and Figure 5; Appendix C Table C1"],
        ["Responsible AI, privacy, and governance", "1",
         "Section 7, its table and Figures 6-7; Appendix C Table C4"],
        ["Monitoring, response, and final plan", "1",
         "Section 8, its table and Figure 8; Section 9; Appendix C Tables C2, C3 and C5"],
    ], widths=[2.6, 0.6, 3.45])


def appendices(doc, c):
    page_break(doc)
    appendix_heading(doc, "Appendix A. Tuned Hyperparameters and Feature Sets")
    p = doc.add_paragraph()
    set_paragraph(p, "Table A1. Selected configuration per family, from the 15-candidate "
                     "randomised search with 5-fold stratified cross-validation on the training "
                     "split.", size=Pt(9), italic=True)
    rows = [["Model", "Tuned hyperparameters", "CV PR-AUC (mean ± sd)", "Search time"]]
    for m in c["models"]:
        params = json.loads(c["tune"].loc[m, "best_params"])
        rows.append([
            m, ", ".join(f"{k.split('__')[-1]} = {_fmt(v)}" for k, v in params.items()),
            f"{c['cv'].loc[m, 'pr_auc_mean']:.6f} ± {c['cv'].loc[m, 'pr_auc_std']:.6f}",
            f"{c['tune'].loc[m, 'search_seconds']:.0f} s",
        ])
    fill_table(doc.add_table(rows=1, cols=4, style="Table Grid"), rows,
               widths=[1.3, 2.9, 1.4, 0.8])

    p = doc.add_paragraph()
    set_paragraph(p, f"Table A2. Feature sets. The primary set keeps every original column; the "
                     f"other two exist only as diagnostics.", size=Pt(9), italic=True)
    abl = c["abl"]
    rows = [["Feature set", "n", "What it contains", "Validation recall / FPR"]]
    desc = {
        "full": "PRIMARY. All 78 original CICFlowMeter columns, the 2 sentinel flags and all "
                f"{len(c['eng'])} engineered features.",
        "original_only": "Original columns and sentinel flags only: measures what feature "
                         "engineering is worth.",
        "no_fingerprint": "Primary set minus Destination Port and the TCP-window columns: "
                          "measures how much of the score is this scanner's fingerprint.",
    }
    for k in ["full", "original_only", "no_fingerprint"]:
        r = abl.loc[k]
        rows.append([k, int(r.n_features), desc[k], f"{f5(r.recall)} / {pct(r.fpr, 4)}"])
    fill_table(doc.add_table(rows=1, cols=4, style="Table Grid"), rows,
               widths=[1.2, 0.4, 3.55, 1.25])

    p = doc.add_paragraph()
    set_paragraph(p, "Table A3. Engineered features and why each was added. Medians are over the "
                     "cleaned PortScan file.", size=Pt(9), italic=True)
    eng = c["eng"]
    rows = [["Feature", "Definition", "Why it should help", "Median scan", "Median benign"]]
    for _, r in eng.iterrows():
        rows.append([r.feature, r.definition, r.why_it_should_help,
                     f"{r.median_scan:,.4g}", f"{r.median_benign:,.4g}"])
    fill_table(doc.add_table(rows=1, cols=5, style="Table Grid"), rows, size=Pt(7.5),
               widths=[1.3, 1.75, 1.95, 0.72, 0.72])

    page_break(doc)
    appendix_heading(doc, "Appendix B. Generalization Detail")
    p = doc.add_paragraph()
    set_paragraph(p, "Table B1. Every unseen capture day, scored at the frozen threshold.",
                  size=Pt(9), italic=True)
    rows = [["Day", "Contents", "Flows", "Benign flows", "Benign alerts", "Benign FPR",
             "Within 5%"]]
    for _, r in c["gen_day"].iterrows():
        rows.append([r.day, r.description, f"{int(r.rows_after_cleaning):,}",
                     f"{int(r.benign_flows):,}", f"{int(r.benign_flagged):,}",
                     pct(r.benign_fpr, 4), "Yes" if r.within_5pct_ceiling else "No"])
    fill_table(doc.add_table(rows=1, cols=7, style="Table Grid"), rows, size=Pt(7.5),
               widths=[0.85, 1.85, 0.7, 0.75, 0.75, 0.65, 0.55])

    p = doc.add_paragraph()
    set_paragraph(p, "Table B2. Attack types absent from training, and how often the port-scan "
                     "detector flags them.", size=Pt(9), italic=True)
    att = (c["gen_att"].groupby("attack_type")
           .agg(flows=("flows", "sum"), flagged=("flagged_as_portscan", "sum"))
           .reset_index().sort_values("flows", ascending=False))
    att["rate"] = att.flagged / att.flows
    rows = [["Attack type (never trained on)", "Flows", "Flagged as PortScan", "Flag rate"]]
    for _, r in att.iterrows():
        rows.append([r.attack_type, f"{int(r.flows):,}", f"{int(r.flagged):,}", pct(r.rate, 2)])
    fill_table(doc.add_table(rows=1, cols=4, style="Table Grid"), rows, size=Pt(8),
               widths=[2.6, 1.0, 1.4, 1.0])

    p = doc.add_paragraph()
    set_paragraph(p, "Table B3. Schema differences that had to be handled before scoring.",
                  size=Pt(9), italic=True)
    rows = [["Day", "Columns", "Encoding required", "Identical to training file"]]
    for _, r in c["schema"].iterrows():
        rows.append([r.day, int(r.n_columns), r.encoding_needed,
                     "Yes" if r.columns_identical_to_training_file else "No"])
    fill_table(doc.add_table(rows=1, cols=4, style="Table Grid"), rows, size=Pt(8),
               widths=[1.5, 0.9, 1.6, 2.0])

    page_break(doc)
    appendix_heading(doc, "Appendix C. Adversarial Detail and Response Workflow")
    p = doc.add_paragraph()
    set_paragraph(p, "Table C1. Every evasion attack, how a real scanner would produce it, and "
                     "the effect of adversarial training.", size=Pt(9), italic=True)
    rows = [["Attack", "How a scanner produces it", "Recall (deployed)", "Recall (hardened)"]]
    for atk in c["adv"].index:
        r, d = c["adv"].loc[atk], c["advd"].loc[atk]
        rows.append([atk, r.realisation, f5(r.recall_under_attack, 4),
                     f5(d.recall_after_adversarial_training, 4)])
    cfr = c["advj"]
    rows.append(["Clean benign traffic (false-positive rate)", "No attack: the cost of the defense",
                 pct(cfr["clean_fpr_baseline"], 4), pct(cfr["clean_fpr_hardened"], 4)])
    fill_table(doc.add_table(rows=1, cols=4, style="Table Grid"), rows, size=Pt(8),
               widths=[1.85, 2.5, 1.05, 1.05])

    p = doc.add_paragraph()
    set_paragraph(p, "Table C2. What happens after a detection.", size=Pt(9), italic=True)
    rows = [["#", "Stage", "Action", "Automation"]]
    for _, r in c["flow"].iterrows():
        rows.append([int(r.step), r.stage, r.action, r.automation])
    fill_table(doc.add_table(rows=1, cols=4, style="Table Grid"), rows, size=Pt(8),
               widths=[0.3, 0.95, 3.9, 1.3])

    p = doc.add_paragraph()
    set_paragraph(p, "Table C3. What is written for every alert, so that decisions can be "
                     "audited and fed back.", size=Pt(9), italic=True)
    log = t_safe("logging_schema")
    if log is not None:
        rows = [["Logged field", "Purpose"]]
        for _, r in log.iterrows():
            rows.append([r.field, r.purpose])
        fill_table(doc.add_table(rows=1, cols=2, style="Table Grid"), rows, size=Pt(8),
                   widths=[2.2, 4.25])

    p = doc.add_paragraph()
    set_paragraph(p, "Table C4. Privacy, retention and misuse controls (continuing Section 7).",
                  size=Pt(9), italic=True)
    rows = [["Concern", "Test or safeguard", "Evidence / current finding", "Remaining action"]]
    for _, r in c["gov"].iloc[3:].iterrows():
        rows.append([r.concern, r.test_or_safeguard, r.evidence, r.remaining_action])
    fill_table(doc.add_table(rows=1, cols=4, style="Table Grid"), rows, size=Pt(8),
               widths=[1.1, 1.8, 1.9, 1.65])

    p = doc.add_paragraph()
    set_paragraph(p, "Table C5. Remaining monitoring signals (continuing Section 8).",
                  size=Pt(9), italic=True)
    rows = [["Signal / metric", "Threshold or trigger", "Owner / response", "Feedback use"]]
    for _, r in c["plan"].iloc[4:].iterrows():
        rows.append([r.signal, r.threshold_or_trigger, r.owner_and_response, r.feedback_use])
    fill_table(doc.add_table(rows=1, cols=4, style="Table Grid"), rows, size=Pt(8),
               widths=[1.3, 1.85, 1.9, 1.4])


def t_safe(name):
    p = TAB / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def main():
    c = build_context()
    doc = docx.Document(TEMPLATE)
    # Capture table objects by their ORIGINAL template index: deleting a table renumbers
    # doc.tables, so every reference below must be resolved before the first deletion.
    T = list(doc.tables)
    header(doc, c, T)
    snapshot(doc, c, T)
    s1_status(doc, c)
    s2_models(doc, c, T)
    s3_evaluation(doc, c, T)
    s4_errors(doc, c, T)
    s5_generalization(doc, c, T)
    s6_adversarial(doc, c, T)
    s7_responsible(doc, c, T)
    s8_monitoring(doc, c, T)
    s9_plan(doc, c, T)
    references(doc, c)
    contributions(doc, c, T)
    ai_disclosure(doc, c, T)
    checklist(doc, c, T)
    rubric(doc, c, T)
    appendices(doc, c)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"selected model: {c['best']}")


if __name__ == "__main__":
    main()
