"""Step 11 (Deliverable 3): Adversarial threat model and evasion/defense experiment.

Threat model
  Asset        the deployed ScanShield classifier and the early warning it provides.
  Attacker     an external reconnaissance actor who assumes a flow-based detector is watching.
               Black box: no access to the model, its parameters, its scores or its training
               data. Full control of their own scanning host.
  Objective    enumerate open ports while keeping every flow below the alert threshold.
  Capability   the features are computed from the attacker's own packets, so the attacker can
               shape timing, payload size, TCP options and handshake completion. The attacker
               cannot touch the defender's data, labels or model (so this is evasion at
               inference time, not poisoning).
  Surface      the four shaping levers below all map to documented scanner options.

Attacks tested (all offline, on saved feature rows from a public research dataset; no live
scanning and no system is touched):
  A1 Timing evasion      nmap -T0/-T1 style pacing: stretch every duration and inter-arrival
                         time by 10x, 100x, 1000x and recompute the dependent rates.
  A2 Payload padding     nmap --data-length: give each probe 24 or 64 bytes of filler.
  A3 Window reshaping    set the initial TCP window to an everyday value instead of the 1024
                         this scanner uses, erasing the tool fingerprint.
  A4 Handshake mimicry   complete the connection (TCP connect scan) so a reply comes back.
  A5 Combined            A1(100x) + A2(24B) + A3 + A4 together.

Each attack perturbs the RAW CICFlowMeter columns and then re-derives every engineered feature,
so the perturbed rows stay internally consistent.

Defense: adversarial training. The defender can generate these shaped flows without ever
meeting the attacker, because they are just traffic-shaping transforms. Perturbed copies of the
TRAINING scan flows are added to the training set and the same model configuration is refit.
Clean performance and per-attack recall are reported before and after.

Outputs:
  tables/adversarial_threat_model.csv, adversarial_results.csv, adversarial_defense.csv
  figures/fig14_adversarial.png
  metrics/adversarial.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.base import clone

from config import PROC_DIR, SPLIT_FILE, MET_DIR, TAB_DIR, TARGET, RANDOM_STATE, TARGET_FPR
from utils import save_table, save_json, save_fig, style, select_model

style()
MODEL_DIR = PROC_DIR / "models"
RNG = np.random.default_rng(RANDOM_STATE)

IAT_COLS = ["Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
            "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
            "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
            "Active Mean", "Active Std", "Active Max", "Active Min",
            "Idle Mean", "Idle Std", "Idle Max", "Idle Min"]
COMMON_WINDOWS = np.array([8192, 29200, 64240, 65535])


def _load_module(stem):
    path = Path(__file__).resolve().parent / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem.replace(".", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def recompute_rates(d):
    """Keep the rate columns consistent with the counts and durations after a perturbation."""
    secs = (d["Flow Duration"].astype("float64") / 1e6).clip(lower=1e-9)
    fwd, bwd = d["Total Fwd Packets"], d["Total Backward Packets"]
    total_bytes = d["Total Length of Fwd Packets"] + d["Total Length of Bwd Packets"]
    d["Flow Bytes/s"] = total_bytes / secs
    d["Flow Packets/s"] = (fwd + bwd) / secs
    d["Fwd Packets/s"] = fwd / secs
    d["Bwd Packets/s"] = bwd / secs
    d["Down/Up Ratio"] = (bwd / fwd.clip(lower=1)).astype("int64")
    return d


def a1_timing(d, factor):
    """Pace the scan: everything that measures time is stretched."""
    d = d.copy()
    d["Flow Duration"] = (d["Flow Duration"] * factor).astype("int64")
    for c in IAT_COLS:
        d[c] = d[c] * factor
    return recompute_rates(d)


def a2_padding(d, nbytes):
    """Give every probe a payload, the way nmap --data-length does."""
    d = d.copy()
    fwd = d["Total Fwd Packets"].clip(lower=1)
    d["Total Length of Fwd Packets"] = d["Total Length of Fwd Packets"] + nbytes * fwd
    for c in ["Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
              "Avg Fwd Segment Size"]:
        d[c] = np.maximum(d[c], nbytes)
    d["Subflow Fwd Bytes"] = d["Subflow Fwd Bytes"] + nbytes * fwd
    d["Max Packet Length"] = np.maximum(d["Max Packet Length"], nbytes)
    d["Min Packet Length"] = np.maximum(d["Min Packet Length"], nbytes)
    d["Packet Length Mean"] = np.maximum(d["Packet Length Mean"], nbytes)
    d["Average Packet Size"] = np.maximum(d["Average Packet Size"], nbytes)
    d["act_data_pkt_fwd"] = d["Total Fwd Packets"]
    return recompute_rates(d)


def a3_window(d):
    """Erase the tool fingerprint: use an ordinary initial TCP window."""
    d = d.copy()
    obs = d["Init_Win_bytes_forward"] >= 0
    d.loc[obs, "Init_Win_bytes_forward"] = RNG.choice(COMMON_WINDOWS, size=int(obs.sum()))
    return d


def a4_handshake(d):
    """Complete the connection so the flow gets an answer, like a TCP connect scan."""
    d = d.copy()
    d["Total Backward Packets"] = d["Total Backward Packets"] + 2
    d["Bwd Header Length"] = d["Bwd Header Length"] + 40
    d["ACK Flag Count"] = d["ACK Flag Count"] + 1
    d["FIN Flag Count"] = d["FIN Flag Count"] + 1
    d["Init_Win_bytes_backward"] = np.where(d["Init_Win_bytes_backward"] < 0, 29200,
                                            d["Init_Win_bytes_backward"])
    return recompute_rates(d)


def a5_combined(d):
    return a4_handshake(a3_window(a2_padding(a1_timing(d, 100), 24)))


ATTACKS = [
    ("A0 No attack (clean scan flows)", lambda d: d.copy(),
     "Reference: the scan flows exactly as captured"),
    ("A1 Timing evasion, 10x slower", lambda d: a1_timing(d, 10),
     "nmap -T2: probes spaced out by an order of magnitude"),
    ("A1 Timing evasion, 100x slower", lambda d: a1_timing(d, 100),
     "nmap -T1: roughly one probe per several seconds"),
    ("A1 Timing evasion, 1000x slower", lambda d: a1_timing(d, 1000),
     "nmap -T0: a scan paced over hours"),
    ("A2 Payload padding, 24 bytes", lambda d: a2_padding(d, 24),
     "nmap --data-length 24: probes stop looking empty"),
    ("A2 Payload padding, 64 bytes", lambda d: a2_padding(d, 64),
     "nmap --data-length 64: probes look like small requests"),
    ("A3 TCP window reshaping", a3_window,
     "Scanner sets an everyday initial window instead of 1024"),
    ("A4 Handshake mimicry (connect scan)", a4_handshake,
     "nmap -sT: the connection is completed so a reply exists"),
    ("A5 Combined (A1 100x + A2 + A3 + A4)", a5_combined,
     "All four levers used together: the strongest attack tested"),
]


def main():
    feat_mod = _load_module("07_feature_engineering")
    fs = json.load(open(MET_DIR / "feature_sets_v2.json"))
    cols = fs["full"]
    engineered = set(fs["engineered"])
    raw_cols = [c for c in cols if c not in engineered]

    df = pd.read_parquet(PROC_DIR / "model_matrix.parquet")
    splits = pd.read_parquet(SPLIT_FILE)
    comp = pd.read_csv(TAB_DIR / "model_comparison.csv")
    best = select_model()
    thr = float(pd.read_csv(TAB_DIR / "threshold_choice.csv").set_index("model")
                .loc[best, "threshold"])
    model = joblib.load(MODEL_DIR / f"{best.lower().replace(' ', '_')}.joblib")
    print(f"attacking {best} at frozen threshold {thr:.6f}")

    m_tr = (splits.strat_split == "train").values
    m_te = (splits.strat_split == "test").values
    test = df.loc[m_te]
    test_scan = test[test[TARGET] == 1]
    test_benign = test[test[TARGET] == 0]
    print(f"test scan flows under attack: {len(test_scan):,}")

    def recall_under(mdl, attack_fn, frame):
        pert = feat_mod.engineer(attack_fn(frame[raw_cols + [TARGET]].copy()))
        s = mdl.predict_proba(pert[cols])[:, 1]
        return float((s >= thr).mean()), s

    # ---- baseline model under every attack -------------------------------------------------
    rows = []
    for name, fn, descr in ATTACKS:
        rec, _ = recall_under(model, fn, test_scan)
        rows.append({"attack": name, "realisation": descr, "recall_under_attack": rec,
                     "evasion_rate": 1 - rec, "scan_flows": len(test_scan)})
        print(f"  {name:42s} recall {rec:.4f}  evasion {1-rec:.4f}")
    base = pd.DataFrame(rows)

    # ---- defense: adversarial training ------------------------------------------------------
    # Harden the SAME model family and configuration that was selected, so the before/after
    # comparison isolates the effect of the augmented data and nothing else.
    print("\n=== defense: adversarial training ===")
    train = df.loc[m_tr]
    train_scan = train[train[TARGET] == 1]
    # Augment with one shaped copy per lever, each on a random third of the training scan flows,
    # so the defender covers the attack surface without simply duplicating the data.
    aug_parts = []
    for fn in [lambda d: a1_timing(d, RNG.choice([10, 100, 1000])),
               lambda d: a2_padding(d, int(RNG.choice([24, 64]))),
               a3_window, a4_handshake, a5_combined]:
        sample = train_scan.sample(frac=0.33, random_state=RANDOM_STATE)
        aug_parts.append(feat_mod.engineer(fn(sample[raw_cols + [TARGET]].copy())))
    aug = pd.concat(aug_parts, ignore_index=True)
    X_aug = pd.concat([train[cols], aug[cols]], ignore_index=True)
    y_aug = np.concatenate([train[TARGET].values, aug[TARGET].values])
    print(f"  training rows {len(train):,} + {len(aug):,} shaped scan copies = {len(X_aug):,}")

    hardened = clone(model)
    hardened.fit(X_aug, y_aug)
    joblib.dump(hardened, MODEL_DIR / f"{best.lower().replace(' ', '_')}_hardened.joblib")

    # clean behaviour must not degrade
    s_ben = hardened.predict_proba(test_benign[cols])[:, 1]
    s_scan = hardened.predict_proba(test_scan[cols])[:, 1]
    hardened_clean_fpr = float((s_ben >= thr).mean())
    hardened_clean_recall = float((s_scan >= thr).mean())
    base_clean_fpr = float((model.predict_proba(test_benign[cols])[:, 1] >= thr).mean())

    def_rows = []
    for name, fn, descr in ATTACKS:
        rec_h, _ = recall_under(hardened, fn, test_scan)
        rec_b = float(base[base.attack == name].recall_under_attack.iloc[0])
        def_rows.append({"attack": name, "recall_baseline_model": rec_b,
                         "recall_after_adversarial_training": rec_h,
                         "recall_recovered": rec_h - rec_b})
        print(f"  {name:42s} {rec_b:.4f} -> {rec_h:.4f}  ({rec_h-rec_b:+.4f})")
    dfn = pd.DataFrame(def_rows)
    dfn.loc[len(dfn)] = {"attack": "Clean benign traffic (false-positive rate)",
                         "recall_baseline_model": base_clean_fpr,
                         "recall_after_adversarial_training": hardened_clean_fpr,
                         "recall_recovered": hardened_clean_fpr - base_clean_fpr}
    save_table(base, "adversarial_results")
    save_table(dfn, "adversarial_defense")

    tm = pd.DataFrame([{
        "asset": "Deployed ScanShield classifier and the early-warning signal it provides",
        "attacker_objective": "Enumerate open ports without any flow crossing the alert threshold",
        "capability": ("Black box: no access to the model, its scores or its training data. "
                       "Full control of their own scanning host, so they can shape probe timing, "
                       "payload length, TCP window and handshake completion."),
        "attack_surface": "The flow features are derived from the attacker's own packets",
        "attack_tested": ("Feature-space evasion by traffic shaping: 5 attacks (timing, payload "
                          "padding, TCP-window reshaping, handshake mimicry, all combined), "
                          "applied offline to held-out scan flows"),
        "defense_tested": ("Adversarial training: the same shaping transforms applied to training "
                           "scan flows and added to the training set, model refit"),
        "environment": ("Offline only, on the public CICIDS2017 research dataset. No live "
                        "scanning, no traffic generated, no system contacted."),
    }])
    save_table(tm, "adversarial_threat_model")

    # ---- figure ------------------------------------------------------------------------------
    plot = dfn[dfn.attack != "Clean benign traffic (false-positive rate)"].copy()
    plot["short"] = plot.attack.str.replace(r"^A\d ", "", regex=True).str.replace(
        r" \(.*\)$", "", regex=True)
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.barh(y + 0.19, plot.recall_baseline_model, height=0.36, color="#c0392b",
            label="Deployed model")
    ax.barh(y - 0.19, plot.recall_after_adversarial_training, height=0.36, color="#2a78b5",
            label="After adversarial training")
    ax.set_yticks(y, plot.short, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0.90, color="gray", ls=":", lw=1.2)
    ax.text(0.90, -0.85, "recall target 0.90", color="gray", fontsize=9, ha="center")
    for i, (b, h) in enumerate(zip(plot.recall_baseline_model, plot.recall_after_adversarial_training)):
        ax.text(b + .006, i + 0.19, f"{b:.3f}", va="center", fontsize=8, color="#c0392b")
        ax.text(h + .006, i - 0.19, f"{h:.3f}", va="center", fontsize=8, color="#2a78b5")
    ax.set_xlim(0, 1.09)
    ax.set_xlabel("Recall on held-out scan flows after the attack is applied")
    ax.set_title("Evasion attacks and the effect of adversarial training\n"
                 f"({len(test_scan):,} held-out scan flows, {best}, frozen threshold)",
                 fontsize=12.5)
    ax.legend(loc="lower right")
    save_fig(fig, "fig14_adversarial")

    save_json({"model": best, "threshold": thr, "test_scan_flows": len(test_scan),
               "baseline": rows, "defense": def_rows,
               "clean_fpr_baseline": base_clean_fpr,
               "clean_fpr_hardened": hardened_clean_fpr,
               "clean_recall_hardened": hardened_clean_recall,
               "augmented_training_rows": int(len(X_aug))}, "adversarial")
    print(f"\nClean FPR {base_clean_fpr:.6f} -> {hardened_clean_fpr:.6f}; "
          f"clean recall after hardening {hardened_clean_recall:.6f}")


if __name__ == "__main__":
    main()
