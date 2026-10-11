"""Run with:  python -m pytest tests   (or: python tests/test_slow_scan.py)"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slow_scan import per_source_window_features, flag_slow_scanners  # noqa: E402

T0 = pd.Timestamp("2026-10-13 00:00")


def synthetic():
    # slow scanner: 1 probe every 10 minutes to a new port, never answered, for 20 hours (120 ports)
    slow = pd.DataFrame({"src_ip": "10.0.0.66", "dst_ip": "10.0.0.5", "dst_port": range(1000, 1120),
                         "timestamp": [T0 + pd.Timedelta(minutes=10 * i) for i in range(120)], "bwd_packets": 0})
    # normal user: 200 web requests to 2 ports, answered
    user = pd.DataFrame({"src_ip": "10.0.0.7", "dst_ip": "10.0.0.80", "dst_port": [80, 443] * 100,
                         "timestamp": [T0 + pd.Timedelta(minutes=5 * i) for i in range(200)], "bwd_packets": 3})
    return pd.concat([slow, user], ignore_index=True)


def test_slow_scan_invisible_in_short_window_but_flagged_over_24h():
    f = per_source_window_features(synthetic())
    s = f[f.src_ip == "10.0.0.66"]
    assert s["distinct_ports_5min"].max() == 1            # each 5-minute window looks harmless
    assert s["distinct_ports_24h"].max() == 120
    flagged = flag_slow_scanners(f)
    assert flagged[f.src_ip == "10.0.0.66"].any()
    assert not flagged[f.src_ip == "10.0.0.7"].any()     # normal user is never flagged


if __name__ == "__main__":
    test_slow_scan_invisible_in_short_window_but_flagged_over_24h(); print("ok")
