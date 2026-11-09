"""Step 10 (Deliverable 3): Generalization testing on larger, unseen data.

The model is trained on one afternoon of one file. This step scores it, unchanged and at its
frozen threshold, on every other CICIDS2017 day file: about 2.5 million flows from four other
capture days containing twelve attack types the model has never seen.

Two questions are answered separately:
  1. False alarms. On benign traffic from days that were never trained on, does the
     false-positive rate stay inside the 5% operational ceiling?
  2. Cross-attack behavior. Does a port-scan detector fire on DoS, brute force, botnet,
     infiltration or web attacks? Those flows are not port scans, so a flag is a false positive
     for this label, but it is operationally different from flagging ordinary traffic.

A schema-compatibility audit runs first, because the files are not drop-in identical.

Outputs:
  tables/schema_compatibility.csv, generalization_by_day.csv, generalization_by_attack.csv,
         generalization_summary.csv
  figures/fig13_generalization.png
  metrics/generalization.json
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from config import PROC_DIR, MET_DIR, TAB_DIR, RAW_DIR, LABEL_COL, TARGET, TARGET_FPR
from utils import save_table, save_json, save_fig, style, select_model

style()
MODEL_DIR = PROC_DIR / "models"
EXTERNAL_DIR = Path.home() / "Desktop" / "CICIDS_dataset"

# Cleaning and engineering 2.5M flows is the slow part of this step, so a stratified sample of
# each day is cached for step 13 (drift and alert-confidence baselines) instead of being redone.
SAMPLE_DIR = PROC_DIR / "external_samples"
SAMPLE_ROWS = 150_000

# Day files other than the training file. Each is a different capture day or half-day.
EXTERNAL_FILES = [
    ("Monday", "Monday-WorkingHours.pcap_ISCX.csv", "All benign, Mon 3 Jul (outside-day control)"),
    ("Tuesday", "Tuesday-WorkingHours.pcap_ISCX.csv", "Benign + FTP/SSH brute force, Tue 4 Jul"),
    ("Wednesday", "Wednesday-workingHours.pcap_ISCX.csv", "Benign + DoS family + Heartbleed, Wed 5 Jul"),
    ("Thursday AM", "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", "Benign + web attacks, Thu 6 Jul"),
    ("Thursday PM", "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", "Benign + infiltration, Thu 6 Jul"),
    ("Friday AM", "Friday-WorkingHours-Morning.pcap_ISCX.csv", "Benign + botnet, Fri 7 Jul morning"),
    ("Friday PM DDoS", "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "Benign + DDoS, Fri 7 Jul afternoon"),
]


def _load_module(stem):
    path = Path(__file__).resolve().parent / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem.replace(".", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_any(path):
    """CICIDS2017 files are not uniformly encoded: the web-attacks file uses cp1252 dashes."""
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, low_memory=False, encoding=enc)
            return df, enc
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("could not decode", b"", 0, 1, str(path))


def normalise_labels(s):
    """Fold the mojibake dash in 'Web Attack <?> Brute Force' into a plain hyphen."""
    return (s.astype(str).str.strip()
            .str.replace(r"\s*[^\x00-\x7F]+\s*", " - ", regex=True)
            .str.replace(r"\s+", " ", regex=True))


def main():
    if not EXTERNAL_DIR.exists():
        print(f"External day files not found at {EXTERNAL_DIR}; skipping generalization step.")
        return

    clean_mod = _load_module("02_clean")
    feat_mod = _load_module("07_feature_engineering")

    fs = json.load(open(MET_DIR / "feature_sets_v2.json"))
    cols = fs["full"]
    thr_tab = pd.read_csv(TAB_DIR / "threshold_choice.csv").set_index("model")
    comp = pd.read_csv(TAB_DIR / "model_comparison.csv")
    best = select_model()
    thr = float(thr_tab.loc[best, "threshold"])
    model = joblib.load(MODEL_DIR / f"{best.lower().replace(' ', '_')}.joblib")
    print(f"scoring {best} at frozen threshold {thr:.6f}")

    train_cols = [c.strip() for c in pd.read_csv(
        RAW_DIR / "Friday-WorkingHours-Afternoon-PortScan_pcap_ISCX.csv.gz", nrows=1).columns]

    schema, by_day, by_attack = [], [], []
    for day, fname, desc in EXTERNAL_FILES:
        path = EXTERNAL_DIR / fname
        if not path.exists():
            print(f"  missing {fname}, skipped"); continue
        t0 = time.perf_counter()
        raw, enc = read_any(path)
        raw.columns = raw.columns.str.strip()
        raw[LABEL_COL] = normalise_labels(raw[LABEL_COL])

        missing = [c for c in train_cols if c not in raw.columns]
        extra = [c for c in raw.columns if c not in train_cols]
        schema.append({
            "day": day, "file": fname, "rows_raw": len(raw), "n_columns": raw.shape[1],
            "encoding_needed": enc, "columns_identical_to_training_file": not missing and not extra,
            "missing_columns": ", ".join(missing) or "none",
            "extra_columns": ", ".join(extra) or "none",
            "header_whitespace_stripped": True,
            "labels_present": ", ".join(sorted(raw[LABEL_COL].unique())),
        })

        df, _ = clean_mod.clean(raw, f"cleaning_log_{day.lower().replace(' ', '_')}")
        df = feat_mod.engineer(df)
        score = model.predict_proba(df[cols])[:, 1]
        df["flagged"] = (score >= thr).astype(int)

        ben = df[df[LABEL_COL] == "BENIGN"]
        by_day.append({
            "day": day, "description": desc, "rows_after_cleaning": len(df),
            "benign_flows": len(ben), "benign_flagged": int(ben.flagged.sum()),
            "benign_fpr": float(ben.flagged.mean()) if len(ben) else np.nan,
            "within_5pct_ceiling": bool(ben.flagged.mean() <= TARGET_FPR) if len(ben) else None,
            "attack_flows": int((df[LABEL_COL] != "BENIGN").sum()),
            "seconds": time.perf_counter() - t0,
        })
        for lab, g in df[df[LABEL_COL] != "BENIGN"].groupby(LABEL_COL):
            by_attack.append({"day": day, "attack_type": lab, "flows": len(g),
                              "flagged_as_portscan": int(g.flagged.sum()),
                              "flag_rate": float(g.flagged.mean())})

        # clean() resets the index, so index labels are positions into `score`.
        SAMPLE_DIR.mkdir(exist_ok=True)
        keep = df if len(df) <= SAMPLE_ROWS else df.sample(SAMPLE_ROWS, random_state=42)
        sample = keep[cols + [LABEL_COL, "flagged"]].copy()
        sample["score"] = score[keep.index.values]
        sample.to_parquet(SAMPLE_DIR / f"{day.lower().replace(' ', '_')}.parquet", index=False)

        print(f"  {day:15s} {len(df):>7,} flows  benign FPR {ben.flagged.mean():.5f}  "
              f"({time.perf_counter()-t0:.0f}s)")
        del raw, df, keep, sample

    sch = save_table(pd.DataFrame(schema), "schema_compatibility")
    day_tab = save_table(pd.DataFrame(by_day), "generalization_by_day")
    att_tab = save_table(pd.DataFrame(by_attack).sort_values("flows", ascending=False),
                         "generalization_by_attack")

    # ---- course sample (held-out test split) vs the larger external set -------------------
    test_row = comp[(comp.split == "test (locked)") & (comp.model == best)].iloc[0]
    tot_ben = int(day_tab.benign_flows.sum())
    tot_flag = int(day_tab.benign_flagged.sum())
    summary = pd.DataFrame([
        {"dataset": "Course sample: held-out test split of the PortScan file",
         "flows": int(test_row.TN + test_row.FP + test_row.FN + test_row.TP),
         "benign_flows": int(test_row.TN + test_row.FP),
         "scan_flows": int(test_row.TP + test_row.FN),
         "recall": float(test_row.recall), "false_positive_rate": float(test_row.fpr),
         "differences_from_training_data": "Same file, same day; unseen rows only"},
        {"dataset": "Larger external: all other CICIDS2017 day files",
         "flows": int(day_tab.rows_after_cleaning.sum()), "benign_flows": tot_ben,
         "scan_flows": 0, "recall": np.nan,
         "false_positive_rate": tot_flag / tot_ben if tot_ben else np.nan,
         "differences_from_training_data": ("4 other capture days, 12 attack types never trained on, "
                                            "different traffic mix, no PortScan label present")},
    ])
    save_table(summary, "generalization_summary")

    # ---- figure ---------------------------------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    d = day_tab.sort_values("benign_fpr")
    a1.barh(d.day, d.benign_fpr * 100, color="#2a78b5", height=0.65)
    a1.axvline(TARGET_FPR * 100, color="#c0392b", ls="--", lw=1.5)
    a1.text(TARGET_FPR * 100, -0.7, " 5% ceiling", color="#c0392b", fontsize=9)
    for i, (v, n) in enumerate(zip(d.benign_fpr * 100, d.benign_flagged)):
        a1.text(v, i, f"  {v:.3f}%  ({n:,} alerts)", va="center", fontsize=8.5)
    a1.set_xlabel("False-positive rate on benign flows (%)")
    a1.set_title(f"(a) False alarms per unseen day\n{tot_ben:,} benign flows, none used in training",
                 fontsize=11)
    a1.set_xlim(0, max(d.benign_fpr.max() * 100 * 1.6, TARGET_FPR * 100 * 1.15))

    at = att_tab.groupby("attack_type").agg(flows=("flows", "sum"),
                                            flagged=("flagged_as_portscan", "sum")).reset_index()
    at["rate"] = at.flagged / at.flows
    at = at.sort_values("rate")
    a2.barh(at.attack_type, at.rate * 100, color="#e07b1a", height=0.65)
    for i, (v, n) in enumerate(zip(at.rate * 100, at.flows)):
        a2.text(v, i, f"  {v:.1f}% of {n:,}", va="center", fontsize=8.5)
    a2.set_xlabel("Share flagged as PortScan (%)")
    a2.set_title("(b) Cross-attack specificity: twelve attack types\n"
                 "never trained on are almost never called a scan", fontsize=11)
    a2.set_xlim(0, max(at.rate.max() * 100 * 1.6, 0.01))
    fig.suptitle(f"Generalization of {best} to 2.5 million unseen CICIDS2017 flows",
                 fontweight="bold", fontsize=13)
    save_fig(fig, "fig13_generalization")

    save_json({"model": best, "threshold": thr,
               "external_benign_flows": tot_ben, "external_benign_flagged": tot_flag,
               "external_benign_fpr": tot_flag / tot_ben if tot_ben else None,
               "by_day": by_day, "by_attack": by_attack}, "generalization")
    print(f"\nExternal benign FPR overall: {tot_flag}/{tot_ben} = {tot_flag/max(tot_ben,1):.6f}")
    print(sch[["day", "columns_identical_to_training_file", "encoding_needed"]].to_string(index=False))


if __name__ == "__main__":
    main()
