"""Compress the two CICIDS2017 CSVs in data/raw/ to .csv.gz (the form committed to git).

Usage: copy the two original CSVs into data/raw/, then run:  python data/prepare_raw.py
"""
import gzip, hashlib, shutil
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"
EXPECTED = {
    "Friday-WorkingHours-Afternoon-PortScan_pcap_ISCX.csv": "ca1824c51bfbb7b3c72290a11be04366ba8815878c6a1cc5c44cb1cee269e99b",
    "Monday-WorkingHours_pcap_ISCX.csv": "852c4beb34eda186f32561fa79df7a0747e92e1a6535b01270820dd9ffe17f34",
}
RAW.mkdir(exist_ok=True)
for name, sha in EXPECTED.items():
    src, dst = RAW / name, RAW / (name + ".gz")
    if dst.exists():
        print(f"ok (already compressed): {dst.name}"); continue
    if not src.exists():
        print(f"MISSING: put {name} in {RAW}"); continue
    h = hashlib.sha256(src.read_bytes()).hexdigest()
    print(f"{name}: sha256 {'matches' if h == sha else 'DIFFERS from the original used in the report'}")
    with open(src, "rb") as fi, gzip.open(dst, "wb", compresslevel=9) as fo:
        shutil.copyfileobj(fi, fo)
    print(f"wrote {dst.name} ({dst.stat().st_size/1e6:.1f} MB)")
