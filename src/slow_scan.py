"""Slow-scan detection features: count distinct destination ports per source over long windows.

Problem (from the project risk review): an attacker can pace a scan, e.g. one probe every few minutes,
so that every single flow looks normal. A per-flow classifier cannot see this, because the evidence is
spread over hours.

Design: aggregate flows per source IP over several look-back windows (5 min, 1 h, 24 h) and compute
  * distinct_ports_<w>   - number of different destination ports contacted
  * distinct_hosts_<w>   - number of different destination hosts contacted
  * failed_ratio_<w>     - share of flows with no reply packets (typical of probes to closed ports)
A source that touches many distinct ports with mostly failed connections over 24 h is flagged even if
each flow is individually unremarkable.

Like the allowlist, this needs Source IP, Destination IP and Timestamp columns, which the
MachineLearningCVE release lacks. It is unit-tested on synthetic flows (tests/test_slow_scan.py) and will
be evaluated on the GeneratedLabelledFlows release.
"""
import numpy as np
import pandas as pd

WINDOWS = {"5min": "5min", "1h": "1h", "24h": "24h"}


def per_source_window_features(flows: pd.DataFrame) -> pd.DataFrame:
    """flows needs: src_ip, dst_ip, dst_port, timestamp, bwd_packets. For each flow, features are computed
    over the same source's flows in the look-back window ending at that flow (inclusive)."""
    out = flows.sort_values(["src_ip", "timestamp"]).reset_index(drop=True)
    for name, win in WINDOWS.items():
        width = pd.Timedelta(win)
        ports, hosts, failed = [], [], []
        for _, g in out.groupby("src_ip", sort=False):
            ts = g["timestamp"].to_numpy()
            starts = np.searchsorted(ts, ts - width, side="right")
            for i, lo in enumerate(starts):
                w = g.iloc[lo:i + 1]
                ports.append(w["dst_port"].nunique())
                hosts.append(w["dst_ip"].nunique())
                failed.append(float((w["bwd_packets"] == 0).mean()))
        out[f"distinct_ports_{name}"] = ports
        out[f"distinct_hosts_{name}"] = hosts
        out[f"failed_ratio_{name}"] = failed
    return out


def flag_slow_scanners(feats: pd.DataFrame, min_ports_24h: int = 50, min_failed: float = 0.8) -> pd.Series:
    """Simple starting rule (to be tuned on labeled data): many distinct ports in 24 h, mostly failed."""
    return (feats["distinct_ports_24h"] >= min_ports_24h) & (feats["failed_ratio_24h"] >= min_failed)
