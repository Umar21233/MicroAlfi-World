"""
build_data_manifest.py -- Compute SHA-256 of every file under
"Colab training results" (read-only raw data) and write/verify
data_manifest.json at the project root.

Usage:
    python scripts/build_data_manifest.py check    # verify against existing manifest
    python scripts/build_data_manifest.py write     # (re)write the manifest -- only
                                                       run this ONCE, before any other
                                                       step of the audit touches anything
"""

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "Colab training results"
MANIFEST_PATH = PROJECT_ROOT / "data_manifest.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan() -> dict:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw data folder not found: {RAW_DIR}")
    files = sorted(p for p in RAW_DIR.rglob("*") if p.is_file())
    manifest = {}
    for p in files:
        rel = p.relative_to(RAW_DIR).as_posix()
        manifest[rel] = {
            "sha256": sha256_of(p),
            "size_bytes": p.stat().st_size,
        }
    return manifest


def write_manifest():
    manifest = scan()
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print(f"Wrote {MANIFEST_PATH} with {len(manifest)} file hashes.")


def check_manifest():
    if not MANIFEST_PATH.exists():
        print(f"FAIL: {MANIFEST_PATH} does not exist.")
        sys.exit(1)
    with open(MANIFEST_PATH) as f:
        expected = json.load(f)
    actual = scan()

    missing = set(expected) - set(actual)
    extra = set(actual) - set(expected)
    changed = [
        rel for rel in (set(expected) & set(actual))
        if expected[rel]["sha256"] != actual[rel]["sha256"]
    ]

    ok = True
    if missing:
        ok = False
        print(f"FAIL: {len(missing)} file(s) missing from raw data (were present at manifest time):")
        for rel in sorted(missing):
            print(f"  - {rel}")
    if extra:
        ok = False
        print(f"FAIL: {len(extra)} new file(s) found in raw data (not in manifest):")
        for rel in sorted(extra):
            print(f"  - {rel}")
    if changed:
        ok = False
        print(f"FAIL: {len(changed)} file(s) changed (sha256 mismatch):")
        for rel in sorted(changed):
            print(f"  - {rel}")

    if ok:
        print(f"PASS: all {len(expected)} raw files unchanged (sha256 match).")
    else:
        sys.exit(1)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "write":
        write_manifest()
    elif mode == "check":
        check_manifest()
    else:
        print("Usage: build_data_manifest.py [write|check]")
        sys.exit(2)
