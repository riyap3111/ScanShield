"""Central configuration: paths, constants and column groups for ScanShield."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
MET_DIR = OUT_DIR / "metrics"
for d in (PROC_DIR, FIG_DIR, TAB_DIR, MET_DIR):
    d.mkdir(parents=True, exist_ok=True)

def _raw(name):
    """Prefer the committed .csv.gz; fall back to a plain .csv if only that is present."""
    gz = RAW_DIR / f"{name}.csv.gz"
    return gz if gz.exists() else RAW_DIR / f"{name}.csv"


PORTSCAN_RAW = _raw("Friday-WorkingHours-Afternoon-PortScan_pcap_ISCX")
MONDAY_RAW = _raw("Monday-WorkingHours_pcap_ISCX")
PORTSCAN_CLEAN = PROC_DIR / "portscan_clean.parquet"
MONDAY_CLEAN = PROC_DIR / "monday_benign_clean.parquet"
SPLIT_FILE = PROC_DIR / "split_indices.parquet"

RANDOM_STATE = 42
LABEL_COL = "Label"
TARGET = "is_scan"
POS_LABEL = "PortScan"
NEG_LABEL = "BENIGN"

# Proposal success criteria
TARGET_RECALL = 0.90
TARGET_FPR = 0.05

# Facts verified by hand before the pipeline was written (asserted in 01)
EXPECTED = {
    "rows": 286_467,
    "cols": 79,
    "PortScan": 158_930,
    "BENIGN": 127_537,
    "dup_rows": 72_353,
}

CONSTANT_COLS = [
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "CWE Flag Count",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
]
# Exact copies of another column (found in 01 audit). Kept twin shown in comment.
DUPLICATE_COLS = [
    "Fwd Header Length.1",   # == Fwd Header Length
    "Subflow Fwd Packets",   # == Total Fwd Packets
    "Subflow Bwd Packets",   # == Total Backward Packets
    "Subflow Fwd Bytes",     # == Total Length of Fwd Packets
    "Subflow Bwd Bytes",     # == Total Length of Bwd Packets
    "Avg Fwd Segment Size",  # == Fwd Packet Length Mean
    "Fwd PSH Flags",         # == SYN Flag Count (known CICFlowMeter bug)
    "ECE Flag Count",        # == RST Flag Count
]
PORT_COL = "Destination Port"

KEY_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Flow Bytes/s", "Fwd Packet Length Max", "Destination Port",
]

COLORS = {"BENIGN": "#2a78b5", "PortScan": "#e07b1a"}
