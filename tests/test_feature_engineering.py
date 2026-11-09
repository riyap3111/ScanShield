"""Run with:  python -m pytest tests   (or: python tests/test_feature_engineering.py)

These tests pin the two properties the Deliverable 3 results depend on:
  * engineer() is a pure function of the raw columns, so recomputing it on the same raw input
    reproduces the stored matrix exactly. The adversarial experiment's no-attack control relies
    on this, and so does scoring external day files.
  * the evasion transforms change what they claim to change, keep the schema intact and
    introduce no inf/NaN.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def _load(stem):
    spec = importlib.util.spec_from_file_location(stem.replace(".", "_"), SRC / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


fe = _load("07_feature_engineering")
ENGINEERED = [n for n, _, _ in fe.ENGINEERED_DOC]


def make_frame():
    """A small hand-built frame with every column engineer() touches."""
    base = {
        "Destination Port": [80, 50000, 445, 443],
        "Flow Duration": [50, 1_000_000, 120, 30_000],
        "Total Fwd Packets": [1, 12, 1, 4],
        "Total Backward Packets": [0, 10, 1, 6],
        "Total Length of Fwd Packets": [0, 900, 0, 300],
        "Total Length of Bwd Packets": [0, 4000, 0, 2500],
        "Fwd Packet Length Max": [0, 300, 0, 180],
        "Fwd Packet Length Min": [0, 0, 0, 20],
        "Fwd Packet Length Mean": [0.0, 75.0, 0.0, 75.0],
        "Bwd Packet Length Max": [0, 1460, 0, 900],
        "Bwd Packet Length Min": [0, 0, 0, 60],
        "Fwd Header Length": [32, 240, 32, 96],
        "Bwd Header Length": [0, 200, 20, 120],
        "Fwd Packets/s": [20000.0, 12.0, 8333.0, 133.0],
        "Bwd Packets/s": [0.0, 10.0, 8333.0, 200.0],
        "Min Packet Length": [0, 0, 0, 20],
        "Max Packet Length": [0, 1460, 0, 900],
        "Packet Length Mean": [0.0, 220.0, 0.0, 310.0],
        "Packet Length Std": [0.0, 400.0, 0.0, 350.0],
        "Average Packet Size": [0.0, 230.0, 0.0, 320.0],
        "Avg Fwd Segment Size": [0.0, 75.0, 0.0, 75.0],
        "Subflow Fwd Bytes": [0, 900, 0, 300],
        "act_data_pkt_fwd": [0, 10, 0, 3],
        "Init_Win_bytes_forward": [1024, 29200, -1, 64240],
        "Init_Win_bytes_backward": [-1, 26880, -1, 65535],
    }
    for f in ["FIN", "SYN", "RST", "PSH", "ACK", "URG", "CWE", "ECE"]:
        base[f"{f} Flag Count"] = [1 if f == "SYN" else 0] * 4
    # Every timing column the evasion transforms stretch, so the fixture matches the real schema.
    timings = {"Flow IAT Mean": [50.0, 90909.0, 120.0, 3000.0],
               "Flow IAT Std": [0.0, 120000.0, 0.0, 4000.0],
               "Active Mean": [0.0, 500.0, 0.0, 200.0],
               "Idle Mean": [0.0, 900000.0, 0.0, 0.0]}
    for col in fe_timing_cols():
        base[col] = timings.get(col, [10.0, 5000.0, 20.0, 800.0])
    return pd.DataFrame(base)


def fe_timing_cols():
    return adv.IAT_COLS


def test_engineer_adds_every_documented_feature():
    out = fe.engineer(make_frame())
    assert [c for c in ENGINEERED if c not in out.columns] == []


def test_engineer_produces_no_inf_or_nan():
    out = fe.engineer(make_frame())
    vals = out[ENGINEERED].to_numpy(dtype="float64")
    assert np.isfinite(vals).all()


def test_engineer_is_deterministic_and_idempotent():
    a = fe.engineer(make_frame())[ENGINEERED]
    b = fe.engineer(make_frame())[ENGINEERED]
    pd.testing.assert_frame_equal(a, b)
    # Re-running on an already-engineered frame must not change the values.
    c = fe.engineer(fe.engineer(make_frame()))[ENGINEERED]
    pd.testing.assert_frame_equal(a, c)


def test_scan_like_row_reads_as_a_probe():
    out = fe.engineer(make_frame())
    probe = out.iloc[0]       # 1 packet out, nothing back, no payload, 50 microseconds
    assert probe.no_response == 1
    assert probe.is_empty_payload == 1
    assert probe.single_fwd_packet == 1
    assert probe.is_short_flow == 1
    assert probe.syn_only == 1
    assert probe.fwd_win_is_1024 == 1
    assert probe.bytes_per_fwd_packet == 0


def test_browsing_like_row_does_not():
    out = fe.engineer(make_frame())
    session = out.iloc[1]     # 12 packets out, 10 back, payload both ways, 1 second
    assert session.no_response == 0
    assert session.is_empty_payload == 0
    assert session.single_fwd_packet == 0
    assert session.is_short_flow == 0
    assert session.bytes_per_fwd_packet > 0


def test_port_class_buckets():
    out = fe.engineer(make_frame())
    # 80 and 443 and 445 are all below 1024, so all three are class 0; 50000 is dynamic.
    assert list(out.port_class) == [0, 2, 0, 0]
    assert list(out.is_well_known_port) == [1, 0, 1, 1]
    assert list(out.is_common_service_port) == [1, 0, 1, 1]


# ----------------------------------------------------------------- evasion transforms
adv = _load("11_adversarial")


def test_timing_attack_stretches_time_and_rescales_rates():
    before = make_frame()
    after = adv.a1_timing(before.copy(), 100)
    assert (after["Flow Duration"] == before["Flow Duration"] * 100).all()
    assert (after["Flow IAT Mean"] == before["Flow IAT Mean"] * 100).all()
    # rates are recomputed from the new duration, so they must fall
    assert (after["Flow Packets/s"] <= before["Fwd Packets/s"] + before["Bwd Packets/s"]).all()


def test_padding_attack_gives_every_probe_a_payload():
    before = make_frame()
    after = adv.a2_padding(before.copy(), 24)
    grew = after["Total Length of Fwd Packets"] - before["Total Length of Fwd Packets"]
    assert (grew == 24 * before["Total Fwd Packets"].clip(lower=1)).all()
    assert (after["Fwd Packet Length Max"] >= 24).all()


def test_window_attack_removes_the_1024_fingerprint():
    after = adv.a3_window(make_frame())
    observed = after["Init_Win_bytes_forward"] >= 0
    assert not (after.loc[observed, "Init_Win_bytes_forward"] == 1024).any()
    # rows where the window was never observed keep their -1 sentinel
    assert (after.loc[~observed, "Init_Win_bytes_forward"] == -1).all()


def test_handshake_attack_creates_a_reply():
    before = make_frame()
    after = adv.a4_handshake(before.copy())
    assert (after["Total Backward Packets"] > before["Total Backward Packets"]).all()
    assert fe.engineer(after)["no_response"].sum() == 0


def test_attacks_preserve_schema_and_stay_finite():
    for name, fn, _ in adv.ATTACKS:
        out = fe.engineer(fn(make_frame()))
        assert [c for c in ENGINEERED if c not in out.columns] == [], name
        assert np.isfinite(out[ENGINEERED].to_numpy(dtype="float64")).all(), name


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
