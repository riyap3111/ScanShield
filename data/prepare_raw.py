"""Compress the two main CICIDS2017 CSVs to the .csv.gz form the pipeline reads.

Usage: put the two original CSVs in data/raw/ (or in ~/Desktop/CICIDS_dataset/), then run
       python data/prepare_raw.py

CIC ships the files as 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'. A dot inside the
stem confuses tooling that splits on extensions, so the pipeline uses '_pcap_ISCX'. Either
spelling is accepted here and the output is always written with the underscore.
"""
import gzip
import hashlib
import shutil
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"
FALLBACK_DIR = Path.home() / "Desktop" / "CICIDS_dataset"

EXPECTED = {
    "Friday-WorkingHours-Afternoon-PortScan_pcap_ISCX.csv":
        "ca1824c51bfbb7b3c72290a11be04366ba8815878c6a1cc5c44cb1cee269e99b",
    "Monday-WorkingHours_pcap_ISCX.csv":
        "852c4beb34eda186f32561fa79df7a0747e92e1a6535b01270820dd9ffe17f34",
}


def find_source(name):
    """Accept '_pcap_ISCX' or CIC's original '.pcap_ISCX', in data/raw/ or the fallback folder."""
    original = name.replace("_pcap_ISCX", ".pcap_ISCX")
    for folder in (RAW, FALLBACK_DIR):
        for candidate in (folder / name, folder / original):
            if candidate.exists():
                return candidate
    return None


def main():
    RAW.mkdir(exist_ok=True)
    for name, sha in EXPECTED.items():
        dst = RAW / (name + ".gz")
        if dst.exists():
            print(f"ok (already compressed): {dst.name}")
            continue
        src = find_source(name)
        if src is None:
            print(f"MISSING: put {name} in {RAW} (see data/README.md)")
            continue
        h = hashlib.sha256(src.read_bytes()).hexdigest()
        state = "matches" if h == sha else "DIFFERS from the file used in the reports"
        print(f"{src.name}: sha256 {state}")
        with open(src, "rb") as fi, gzip.open(dst, "wb", compresslevel=9) as fo:
            shutil.copyfileobj(fi, fo)
        print(f"wrote {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
