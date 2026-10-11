"""Run with:  python -m pytest tests   (or: python tests/test_allowlist.py)"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from allowlist import load_allowlist, apply_allowlist, match  # noqa: E402

E = load_allowlist()
SAT_2AM = datetime(2026, 10, 10, 2, 0)    # Saturday
TUE_230 = datetime(2026, 10, 13, 2, 30)   # Tuesday
TUE_NOON = datetime(2026, 10, 13, 12, 0)


def test_nessus_in_window_is_suppressed():
    assert match("192.0.2.10", 8080, SAT_2AM, E).ticket == "CHG-1042"


def test_nessus_outside_window_still_alerts():
    assert match("192.0.2.10", 8080, TUE_NOON, E) is None


def test_inventory_sweep_only_on_approved_ports():
    assert match("192.0.2.20", 22, TUE_230, E) is not None
    assert match("192.0.2.20", 31337, TUE_230, E) is None      # unapproved port -> alert


def test_unknown_source_always_alerts():
    assert match("203.0.113.5", 22, SAT_2AM, E) is None


def test_cidr_block_and_expiry():
    assert match("198.51.100.7", 443, TUE_NOON, E) is not None
    assert match("198.51.100.7", 443, datetime(2028, 1, 4, 12, 0), E) is None   # entry expired


def test_apply_allowlist_keeps_audit_trail():
    alerts = pd.DataFrame({"src_ip": ["192.0.2.10", "203.0.113.5"], "dst_port": [445, 445],
                           "timestamp": [SAT_2AM, SAT_2AM]})
    out = apply_allowlist(alerts, E)
    assert list(out.status) == ["suppressed", "alert"]
    assert len(out) == 2 and out.allowlist_entry.iloc[0].startswith("Weekly Nessus")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
