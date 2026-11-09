"""Make XGBoost importable on macOS without Homebrew.

The XGBoost wheel links against an OpenMP runtime (libomp.dylib) that Apple does not ship and
that is normally installed with `brew install libomp`. Where Homebrew is unavailable, dropping
the same library somewhere readable and loading it into the process before XGBoost is imported
satisfies the dependency: dyld then resolves XGBoost's @rpath/libomp.dylib against the image
that is already loaded.

Install the library once with:  python src/install_libomp.py

load_xgboost() returns the module, or None with a short reason if it cannot be loaded, so the
pipeline degrades to the scikit-learn models rather than failing outright.
"""
import ctypes
import os
from pathlib import Path

# Searched in order; the first that exists is loaded.
LIBOMP_CANDIDATES = [
    Path.home() / ".local" / "lib" / "libomp.dylib",
    Path("/opt/homebrew/opt/libomp/lib/libomp.dylib"),
    Path("/usr/local/opt/libomp/lib/libomp.dylib"),
    Path("/opt/local/lib/libomp/libomp.dylib"),
]

_reason = None


def preload_libomp():
    """Load libomp into the process. Returns the path used, or None."""
    env = os.environ.get("SCANSHIELD_LIBOMP")
    candidates = ([Path(env)] if env else []) + LIBOMP_CANDIDATES
    for path in candidates:
        if path.exists():
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                return path
            except OSError:
                continue
    return None


def load_xgboost():
    """Import xgboost, preloading libomp first. Returns (module, note) or (None, reason)."""
    global _reason
    path = preload_libomp()
    try:
        import xgboost  # noqa: F401
    except Exception as exc:                      # ImportError, OSError from dyld, ...
        _reason = (f"xgboost unavailable ({type(exc).__name__}: {str(exc).splitlines()[0][:120]})"
                   + ("" if path else "; no libomp.dylib found, run python src/install_libomp.py"))
        return None, _reason
    return xgboost, f"libomp loaded from {path}" if path else "libomp already present"


if __name__ == "__main__":
    mod, note = load_xgboost()
    print(f"xgboost {mod.__version__} — {note}" if mod else note)
