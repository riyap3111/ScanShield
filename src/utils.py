"""Shared helpers: loading, saving tables, plot styling, feature groups and meanings."""
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import TAB_DIR, MET_DIR, FIG_DIR


def load_raw(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def save_table(df, name, index=False):
    """Save a table as CSV plus a Markdown copy."""
    df.to_csv(TAB_DIR / f"{name}.csv", index=index)
    with open(TAB_DIR / f"{name}.md", "w") as f:
        f.write(df.to_markdown(index=index, floatfmt=".4g"))
    return df


def save_json(obj, name):
    with open(MET_DIR / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def style():
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 300, "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    })


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


class Timer:
    def __enter__(self):
        self.t = time.perf_counter(); return self
    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t


def signed_log1p(X):
    """Compress heavy tails while keeping sign (and the -1 window sentinel) meaningful.

    This lives in utils, not in the training script, because it is pickled inside the logistic
    regression pipeline: a function defined in __main__ cannot be unpickled by any other script.
    """
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


def select_model(tab_dir=None):
    """The one place the selected model is decided, so every step agrees.

    Selection is made on the VALIDATION split, never on test: the test split is scored once,
    after both the threshold and the model choice are already fixed. Among models that meet the
    operational ceiling, the highest F1 wins, with PR-AUC as the tie-break.
    """
    from config import TAB_DIR, TARGET_FPR
    tab_dir = tab_dir or TAB_DIR
    comp = pd.read_csv(tab_dir / "model_comparison.csv")
    val = comp[comp.split == "validation"]
    eligible = val[val.fpr <= TARGET_FPR]
    if eligible.empty:
        eligible = val
    return eligible.sort_values(["f1", "pr_auc"], ascending=False).iloc[0]["model"]


def feature_group(col):
    c = col.lower()
    if col == "Label": return "Target"
    if col == "Destination Port": return "Port"
    if "bulk" in c: return "Bulk"
    if "flag" in c: return "Flags"
    if "init_win" in c: return "TCP window"
    if c.startswith(("active", "idle")): return "Active/Idle"
    if "iat" in c or c == "flow duration": return "Timing"
    if "/s" in c: return "Rates"
    if "length" in c or "size" in c or "bytes" in c or "variance" in c: return "Size"
    if "packets" in c or "act_data" in c or "ratio" in c: return "Counts"
    if "seg_size" in c: return "Size"
    return "Other"


MEANINGS = {
    "Destination Port": "TCP/UDP port number the flow was sent to (the 'door' being knocked on)",
    "Flow Duration": "Time from first to last packet of the flow (microseconds)",
    "Total Fwd Packets": "Packets sent from initiator to responder",
    "Total Backward Packets": "Packets sent from responder back to initiator",
    "Total Length of Fwd Packets": "Total payload bytes in forward direction",
    "Total Length of Bwd Packets": "Total payload bytes in backward direction",
    "Flow Bytes/s": "Bytes per second over the flow",
    "Flow Packets/s": "Packets per second over the flow",
    "Fwd Packets/s": "Forward packets per second",
    "Bwd Packets/s": "Backward packets per second",
    "Min Packet Length": "Smallest packet payload in the flow (bytes)",
    "Max Packet Length": "Largest packet payload in the flow (bytes)",
    "Packet Length Mean": "Mean packet payload size (bytes)",
    "Packet Length Std": "Std. dev. of packet payload size",
    "Packet Length Variance": "Variance of packet payload size",
    "Down/Up Ratio": "Ratio of backward to forward packets",
    "Average Packet Size": "Mean size of all packets in the flow",
    "Avg Fwd Segment Size": "Mean forward segment size",
    "Avg Bwd Segment Size": "Mean backward segment size",
    "Fwd Header Length": "Total bytes of headers in forward packets",
    "Bwd Header Length": "Total bytes of headers in backward packets",
    "Fwd Header Length.1": "Exact duplicate of Fwd Header Length (CICFlowMeter bug)",
    "Fwd PSH Flags": "Times PSH flag set in forward packets (identical to SYN Flag Count in this file)",
    "ECE Flag Count": "Packets with ECE flag (identical to RST Flag Count in this file)",
    "Subflow Fwd Packets": "Mean forward packets per sub-flow",
    "Subflow Fwd Bytes": "Mean forward bytes per sub-flow",
    "Subflow Bwd Packets": "Mean backward packets per sub-flow",
    "Subflow Bwd Bytes": "Mean backward bytes per sub-flow",
    "Init_Win_bytes_forward": "Initial TCP window size sent by initiator (-1 = not observed)",
    "Init_Win_bytes_backward": "Initial TCP window size sent by responder (-1 = not observed)",
    "act_data_pkt_fwd": "Forward packets carrying at least 1 byte of payload",
    "min_seg_size_forward": "Smallest forward segment size observed",
    "Label": "Ground-truth class from CIC: BENIGN or PortScan",
}


def meaning(col):
    if col in MEANINGS: return MEANINGS[col]
    c = col
    for d, word in (("Fwd", "forward"), ("Bwd", "backward")):
        c = c.replace(d, word)
    if "Flag" in col and "Count" in col:
        return f"Number of packets with the TCP {col.split()[0]} flag set"
    if "PSH Flags" in col or "URG Flags" in col:
        return f"Times the {col.split()[1]} flag was set in {('forward' if 'Fwd' in col else 'backward')} packets"
    if "Bulk" in col:
        return f"Bulk-transfer statistic: {c.lower()}"
    if "IAT" in col:
        stat = col.split()[-1].lower()
        direction = "flow" if col.startswith("Flow") else ("forward" if "Fwd" in col else "backward")
        return f"Inter-arrival time between packets ({direction}), {stat} (microseconds)"
    if "Packet Length" in col:
        stat = col.split()[-1].lower()
        return f"{('Forward' if 'Fwd' in col else 'Backward')} packet payload size, {stat} (bytes)"
    if col.startswith(("Active", "Idle")):
        a, s = col.split()
        return f"Time flow was {a.lower()} before going {'idle' if a=='Active' else 'active'}, {s.lower()} (microseconds)"
    return c
