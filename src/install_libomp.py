"""Install the OpenMP runtime XGBoost needs, on a Mac without Homebrew.

XGBoost's macOS wheel links against libomp.dylib, which Apple does not ship. The usual fix is
`brew install libomp`; this script is the fallback when Homebrew is not installed. It downloads
the llvm-openmp package from conda-forge, verifies its SHA256 against the value the project was
built with, and copies libomp.dylib to ~/.local/lib, where src/xgb_support.py looks for it.
Nothing outside ~/.local is touched and no binary is modified.

Usage:  python src/install_libomp.py
        python src/install_libomp.py --check     # report status only
"""
import argparse
import hashlib
import io
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

PKG_URL = ("https://conda.anaconda.org/conda-forge/osx-arm64/"
           "llvm-openmp-23.1.5-hdb3d66b_0.conda")
PKG_SHA256 = "7a68977e62bd0c967509a7a01a25c215bb8027071dafe03aa1976fd8ade94013"
DYLIB_SHA256 = "4395efb4ce9ee38b4ef0ff029150e654ba7b9b0942676584b48071757680dfb2"
INNER = "pkg-llvm-openmp-23.1.5-hdb3d66b_0.tar.zst"
DEST = Path.home() / ".local" / "lib" / "libomp.dylib"


def status():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from xgb_support import load_xgboost, preload_libomp
    found = preload_libomp()
    mod, note = load_xgboost()
    print(f"libomp:  {found or 'not found'}")
    print(f"xgboost: {mod.__version__ + ' — ' + note if mod else note}")
    return mod is not None


def install():
    if DEST.exists():
        print(f"already installed: {DEST}")
        return
    print(f"downloading {PKG_URL}")
    blob = urllib.request.urlopen(PKG_URL, timeout=120).read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != PKG_SHA256:
        sys.exit(f"sha256 mismatch\n  expected {PKG_SHA256}\n  got      {digest}\n"
                 f"Refusing to install a package that is not the one this project was built "
                 f"with. conda-forge may have republished it; verify before changing the hash.")
    print(f"sha256 verified: {digest}")
    try:
        import zstandard
    except ImportError:
        sys.exit("needs the 'zstandard' package: pip install zstandard")
    inner = zipfile.ZipFile(io.BytesIO(blob)).read(INNER)
    data = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(inner)).read()
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        member = tar.extractfile("lib/libomp.dylib")
        if member is None:
            sys.exit("lib/libomp.dylib missing from the package")
        payload = member.read()
    lib_digest = hashlib.sha256(payload).hexdigest()
    if lib_digest != DYLIB_SHA256:
        sys.exit(f"libomp.dylib sha256 mismatch: expected {DYLIB_SHA256}, got {lib_digest}")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_bytes(payload)
    DEST.chmod(0o755)
    print(f"installed {DEST} ({DEST.stat().st_size / 1e3:.0f} KB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report status without installing")
    args = ap.parse_args()
    if args.check:
        sys.exit(0 if status() else 1)
    install()
    status()


if __name__ == "__main__":
    main()
