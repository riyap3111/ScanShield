"""Step 7 (Deliverable 3): Feature engineering.

Every original CICFlowMeter column is kept. This step only ADDS derived features that express
scan behaviour as ratios and indicators the raw counts do not state directly (bytes per packet,
reply ratio, flag density, port class, burstiness). Nothing is removed.

All engineered features are deterministic row-wise functions of the raw columns, so they can be
computed identically on any CICIDS2017 day file and introduce no train/test leakage.

Feature sets written to metrics/feature_sets_v2.json:
  full            PRIMARY. Every original column + sentinel flags + every engineered feature.
  original_only   Original columns + sentinel flags, no engineered features (ablation: does
                  engineering help?).
  no_fingerprint  full minus Destination Port and TCP-window columns (robustness: how much of
                  the score comes from this scanner's tool fingerprint?).

Outputs:
  data/processed/model_matrix.parquet      (PortScan file, engineered)
  data/processed/monday_matrix.parquet     (Monday benign, engineered)
  metrics/feature_sets_v2.json
  tables/engineered_features.csv
"""
import numpy as np
import pandas as pd

from config import (PORTSCAN_CLEAN, MONDAY_CLEAN, PROC_DIR, LABEL_COL, TARGET, PORT_COL)
from utils import save_table, save_json

MODEL_MATRIX = PROC_DIR / "model_matrix.parquet"
MONDAY_MATRIX = PROC_DIR / "monday_matrix.parquet"

NON_FEATURES = {LABEL_COL, TARGET, "raw_row"}

# Ports a normal client contacts constantly; a scanner sweeps far beyond them.
COMMON_SERVICE_PORTS = {80, 443, 53, 22, 21, 25, 110, 143, 445, 139, 3389, 3306, 8080, 993, 995}

# Columns that encode THIS scanner's tooling rather than scanning behaviour in general.
FINGERPRINT_COLS = [
    PORT_COL, "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "init_win_fwd_missing", "init_win_bwd_missing",
]
FINGERPRINT_ENGINEERED = [
    "is_well_known_port", "is_common_service_port", "port_class",
    "fwd_win_is_1024", "win_ratio",
]

# name -> (short description, why it should help)
ENGINEERED_DOC = [
    ("total_packets", "Total Fwd Packets + Total Backward Packets", "Overall flow size in one number"),
    ("total_bytes", "Forward + backward payload bytes", "Overall payload volume in one number"),
    ("bytes_per_fwd_packet", "Forward payload bytes / forward packets", "A probe carries no payload; a real request does"),
    ("bytes_per_bwd_packet", "Backward payload bytes / backward packets", "Separates an empty refusal from a real reply"),
    ("fwd_bwd_packet_ratio", "Forward packets / (backward packets + 1)", "Scans talk and are not answered"),
    ("download_ratio", "Backward bytes / (forward bytes + 1)", "Normal clients mostly receive data"),
    ("no_response", "1 if the responder sent no packet at all", "A closed or filtered port returns nothing"),
    ("is_empty_payload", "1 if the flow carried zero payload bytes", "Probes carry no data in either direction"),
    ("single_fwd_packet", "1 if at most one forward packet was sent", "A SYN probe is one packet"),
    ("is_short_flow", "1 if Flow Duration <= 1000 microseconds", "Probes finish in microseconds"),
    ("duration_per_packet", "Flow Duration / total packets", "Normalises duration by how much was sent"),
    ("flag_total", "Sum of the eight TCP flag counters", "How much TCP signalling the flow used"),
    ("flags_per_packet", "flag_total / total packets", "A probe is almost pure signalling, no data"),
    ("syn_only", "1 if SYN set and no ACK, FIN or PSH", "Signature of a half-open SYN scan"),
    ("rst_present", "1 if any RST flag was seen", "A closed port answers with RST"),
    ("header_bytes_per_fwd_packet", "Forward header bytes / forward packets", "Header-only traffic has no payload to amortise"),
    ("header_to_payload_ratio", "Total header bytes / (total payload bytes + 1)", "Probes are nearly all header"),
    ("win_ratio", "Forward TCP window / (backward window + 1)", "An unanswered handshake leaves this lopsided"),
    ("fwd_win_is_1024", "1 if the initial forward TCP window is exactly 1024", "Value this scanner's SYN probes use"),
    ("iat_burstiness", "Flow IAT Std / (Flow IAT Mean + 1)", "Automated probing is evenly paced, humans are bursty"),
    ("active_idle_ratio", "Active Mean / (Idle Mean + 1)", "Probes are active briefly and never idle"),
    ("pkt_len_cv", "Packet Length Std / (Packet Length Mean + 1)", "Probes are uniform in size, real traffic varies"),
    ("fwd_pkt_len_range", "Fwd Packet Length Max - Fwd Packet Length Min", "Zero when every forward packet is identical"),
    ("bwd_pkt_len_range", "Bwd Packet Length Max - Bwd Packet Length Min", "Zero when there is no varied reply"),
    ("pkts_per_sec_ratio", "Fwd Packets/s / (Bwd Packets/s + 1)", "Rate-based version of the reply imbalance"),
    ("port_class", "0 well-known (<1024), 1 registered, 2 dynamic (>=49152)", "Coarse port grouping that survives a new port list"),
    ("is_well_known_port", "1 if Destination Port < 1024", "Most benign traffic goes to a handful of low ports"),
    ("is_common_service_port", "1 if the port is one of 15 everyday services", "Scanners spend most probes off this list"),
]


def _safe_div(a, b):
    return np.asarray(a, dtype="float64") / np.asarray(b, dtype="float64")


def engineer(df):
    """Add every engineered feature. Pure row-wise arithmetic; safe on any CICIDS2017 day file."""
    d = df
    fwd_pkts = d["Total Fwd Packets"].clip(lower=1)
    bwd_pkts = d["Total Backward Packets"]
    fwd_bytes = d["Total Length of Fwd Packets"]
    bwd_bytes = d["Total Length of Bwd Packets"]

    d["total_packets"] = d["Total Fwd Packets"] + bwd_pkts
    d["total_bytes"] = fwd_bytes + bwd_bytes
    d["bytes_per_fwd_packet"] = _safe_div(fwd_bytes, fwd_pkts)
    d["bytes_per_bwd_packet"] = _safe_div(bwd_bytes, bwd_pkts.clip(lower=1))
    d["fwd_bwd_packet_ratio"] = _safe_div(d["Total Fwd Packets"], bwd_pkts + 1)
    d["download_ratio"] = _safe_div(bwd_bytes, fwd_bytes + 1)
    d["no_response"] = (bwd_pkts == 0).astype(np.int8)
    d["is_empty_payload"] = (d["total_bytes"] == 0).astype(np.int8)
    d["single_fwd_packet"] = (d["Total Fwd Packets"] <= 1).astype(np.int8)
    d["is_short_flow"] = (d["Flow Duration"] <= 1000).astype(np.int8)
    d["duration_per_packet"] = _safe_div(d["Flow Duration"], d["total_packets"].clip(lower=1))

    flags = ["FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
             "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count"]
    d["flag_total"] = d[flags].sum(axis=1)
    d["flags_per_packet"] = _safe_div(d["flag_total"], d["total_packets"].clip(lower=1))
    d["syn_only"] = ((d["SYN Flag Count"] > 0) & (d["ACK Flag Count"] == 0)
                     & (d["FIN Flag Count"] == 0) & (d["PSH Flag Count"] == 0)).astype(np.int8)
    d["rst_present"] = (d["RST Flag Count"] > 0).astype(np.int8)

    d["header_bytes_per_fwd_packet"] = _safe_div(d["Fwd Header Length"], fwd_pkts)
    d["header_to_payload_ratio"] = _safe_div(d["Fwd Header Length"] + d["Bwd Header Length"],
                                             d["total_bytes"] + 1)

    # Init_Win_bytes_* use -1 for "not observed"; treat that as 0 width for the ratio only.
    win_f = d["Init_Win_bytes_forward"].clip(lower=0)
    win_b = d["Init_Win_bytes_backward"].clip(lower=0)
    d["win_ratio"] = _safe_div(win_f, win_b + 1)
    d["fwd_win_is_1024"] = (d["Init_Win_bytes_forward"] == 1024).astype(np.int8)

    d["iat_burstiness"] = _safe_div(d["Flow IAT Std"], d["Flow IAT Mean"].abs() + 1)
    d["active_idle_ratio"] = _safe_div(d["Active Mean"], d["Idle Mean"] + 1)
    d["pkt_len_cv"] = _safe_div(d["Packet Length Std"], d["Packet Length Mean"] + 1)
    d["fwd_pkt_len_range"] = d["Fwd Packet Length Max"] - d["Fwd Packet Length Min"]
    d["bwd_pkt_len_range"] = d["Bwd Packet Length Max"] - d["Bwd Packet Length Min"]
    d["pkts_per_sec_ratio"] = _safe_div(d["Fwd Packets/s"], d["Bwd Packets/s"] + 1)

    port = d[PORT_COL]
    d["port_class"] = np.select([port < 1024, port < 49152], [0, 1], default=2).astype(np.int8)
    d["is_well_known_port"] = (port < 1024).astype(np.int8)
    d["is_common_service_port"] = port.isin(COMMON_SERVICE_PORTS).astype(np.int8)

    # Engineering divides by clipped denominators, so no new inf should appear; guard anyway.
    eng = [n for n, _, _ in ENGINEERED_DOC]
    d[eng] = d[eng].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return d


def feature_sets(df):
    engineered = [n for n, _, _ in ENGINEERED_DOC]
    full = [c for c in df.columns if c not in NON_FEATURES]
    original_only = [c for c in full if c not in engineered]
    drop = set(FINGERPRINT_COLS) | set(FINGERPRINT_ENGINEERED)
    no_fingerprint = [c for c in full if c not in drop]
    return {
        "full": full,
        "original_only": original_only,
        "no_fingerprint": no_fingerprint,
        "engineered": engineered,
        "n_full": len(full),
        "n_original_only": len(original_only),
        "n_no_fingerprint": len(no_fingerprint),
        "note": ("'full' is the primary set and keeps every original CICFlowMeter column. "
                 "'original_only' and 'no_fingerprint' are diagnostic ablations only."),
    }


def main():
    ps = engineer(pd.read_parquet(PORTSCAN_CLEAN))
    ps.to_parquet(MODEL_MATRIX, index=False)
    mon = engineer(pd.read_parquet(MONDAY_CLEAN))
    mon.to_parquet(MONDAY_MATRIX, index=False)

    sets = feature_sets(ps)
    save_json(sets, "feature_sets_v2")

    doc = pd.DataFrame(ENGINEERED_DOC, columns=["feature", "definition", "why_it_should_help"])
    scan, ben = ps[ps[TARGET] == 1], ps[ps[TARGET] == 0]
    doc["median_scan"] = [scan[f].median() for f in doc.feature]
    doc["median_benign"] = [ben[f].median() for f in doc.feature]
    save_table(doc, "engineered_features")

    print(f"model matrix {ps.shape}, monday matrix {mon.shape}")
    print(f"full={sets['n_full']} (incl. {len(sets['engineered'])} engineered), "
          f"original_only={sets['n_original_only']}, no_fingerprint={sets['n_no_fingerprint']}")
    print(doc[["feature", "median_scan", "median_benign"]].to_string(index=False))


if __name__ == "__main__":
    main()
