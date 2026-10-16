"""
build_results_verified.py -- Steps 1 & 2 of the verified-data rebuild.

Reads ONLY from "Colab training results" (never writes there). Validates
every *_models.zip's config.txt against the known stem->(algo, world,
reward_mode, drift) mapping, validates each run's learning_curve.csv has
50 rows ending at timestep 500000, then builds:

    results_verified/<phase>/<run_id>/
        config.txt, learning_curve.csv, training_time.txt, model.zip
        + eval.csv, eval_summary.csv, coverage_trajectory.csv
          (ONLY for the runs whose eval-zip ownership is unambiguous --
           Phase 1, Phase 2, and the turbidity_delta extension; Phase 3's
           eval attribution is resolved later in scripts/reevaluate_all.py)

    results_verified/PROVENANCE.csv
        run_id, file, source_zip, origin, sha256

Usage:
    python scripts/build_results_verified.py            # validate + build
    python scripts/build_results_verified.py --dry-run   # validate only
"""

import csv
import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_naming import canonical_run_id, build_run_name  # noqa: E402

RAW_DIR = PROJECT_ROOT / "Colab training results"
OUT_DIR = PROJECT_ROOT / "results_verified"

# stem-prefix (before "_seedNN") -> (algo, world, reward_mode, drift)
PREFIX_MAP = {
    "a2c":                  ("a2c", "hard",   "multi_objective", "drifts"),
    "dqn":                  ("dqn", "hard",   "multi_objective", "drifts"),
    "ppo":                  ("ppo", "hard",   "multi_objective", "drifts"),
    "dqn_biomass":          ("dqn", "hard",   "biomass_only",    "drifts"),
    "dqn_efficiency":       ("dqn", "hard",   "efficiency",      "drifts"),
    "dqn_sustainability":   ("dqn", "hard",   "sustainability",  "drifts"),
    "dqn_turbidity":        ("dqn", "hard",   "turbidity",       "drifts"),
    "dqn_turbdelta":        ("dqn", "hard",   "turbidity_delta", "drifts"),
    "dqn_easy_drifts":      ("dqn", "easy",   "multi_objective", "drifts"),
    "dqn_easy_nodrifts":    ("dqn", "easy",   "multi_objective", "nodrifts"),
    "dqn_medium_drifts":    ("dqn", "medium", "multi_objective", "drifts"),
    "dqn_medium_nodrifts":  ("dqn", "medium", "multi_objective", "nodrifts"),
    "dqn_hard_nodrifts":    ("dqn", "hard",   "multi_objective", "nodrifts"),
}

STEM_RE = re.compile(r"^(?P<prefix>.+)_seed(?P<seed>\d+)_(?P<kind>models|results)\.zip$")

# Eval zips whose ownership is unambiguous (stated by the user directly).
# Each maps to the set of (world, reward_mode) pairs it holds (always hard,
# always drifts) -- i.e. everything EXCEPT Phase 3's 18 models.
UNAMBIGUOUS_EVAL_ZIPS = {
    "phase1_eval_results.zip":        {("hard", "multi_objective")},   # a2c/dqn/ppo all hard+drifts
    "Phase 2/phase2_eval_results.zip": {("hard", "biomass_only"), ("hard", "efficiency"),
                                         ("hard", "sustainability"), ("hard", "turbidity")},
    "Phase 2/turbidity_delta_eval_results.zip": {("hard", "turbidity_delta")},
}

EVAL_SUFFIXES = ["_eval.csv", "_eval_summary.csv", "_coverage_trajectory.csv"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def phase_of(zip_path: Path) -> str:
    rel = zip_path.relative_to(RAW_DIR)
    parts = rel.parts
    if len(parts) > 1 and parts[0] == "Phase 2":
        return "phase2"
    if len(parts) > 1 and parts[0] == "Phase 3":
        return "phase3"
    return "phase1"


def find_models_zips():
    return sorted(RAW_DIR.rglob("*_models.zip"))


def parse_stem(zip_path: Path):
    m = STEM_RE.match(zip_path.name)
    if not m:
        return None
    return m.group("prefix"), int(m.group("seed")), m.group("kind")


def validate_all(verbose=True):
    """Step 1: validate every models.zip's config.txt against the mapping,
    and every run's learning_curve.csv against the 50-row/500000-step rule.
    Returns (runs, errors) where runs is a list of validated run dicts and
    errors is a list of human-readable problem strings (empty = all good)."""
    runs = []
    errors = []
    seen_run_ids = set()

    for models_zip in find_models_zips():
        parsed = parse_stem(models_zip)
        if parsed is None:
            errors.append(f"UNPARSEABLE stem: {models_zip}")
            continue
        prefix, seed, kind = parsed
        if prefix not in PREFIX_MAP:
            errors.append(f"UNKNOWN prefix {prefix!r} (from {models_zip})")
            continue
        algo, world, reward_mode, drift = PREFIX_MAP[prefix]
        no_drifts = (drift == "nodrifts")
        phase = phase_of(models_zip)

        # 1. Validate config.txt inside models.zip
        with zipfile.ZipFile(models_zip) as z:
            names = z.namelist()
            if "config.txt" not in names:
                errors.append(f"MISSING config.txt in {models_zip}")
                continue
            if "model.zip" not in names:
                errors.append(f"MISSING model.zip in {models_zip}")
                continue
            config_text = z.read("config.txt").decode("utf-8")
            model_bytes = z.read("model.zip")

        config = {}
        for line in config_text.strip().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                config[k.strip()] = v.strip()

        expected = {
            "algo": algo, "world": world, "reward_mode": reward_mode,
            "seed": str(seed), "no_drifts": str(no_drifts),
        }
        mismatches = []
        for k, v_expected in expected.items():
            v_actual = config.get(k)
            if v_actual != v_expected:
                mismatches.append(f"{k}: expected {v_expected!r}, config.txt has {v_actual!r}")
        if "timesteps" not in config:
            mismatches.append("timesteps: missing from config.txt")

        if mismatches:
            errors.append(
                f"CONFIG MISMATCH in {models_zip} "
                f"(mapped to {algo}/{world}/{reward_mode}/{drift}/seed{seed}): "
                + "; ".join(mismatches)
            )
            continue

        # 2. Locate sibling results.zip, validate learning_curve.csv
        results_zip = models_zip.parent / models_zip.name.replace("_models.zip", "_results.zip")
        if not results_zip.exists():
            errors.append(f"MISSING sibling results.zip for {models_zip}")
            continue

        old_name = build_run_name(algo, world, reward_mode, seed, style="old")
        with zipfile.ZipFile(results_zip) as z:
            rnames = set(z.namelist())
            lc_name = f"{old_name}_learning_curve.csv"
            tt_name = f"{old_name}_training_time.txt"
            if lc_name not in rnames:
                errors.append(f"MISSING {lc_name} in {results_zip}")
                continue
            if tt_name not in rnames:
                errors.append(f"MISSING {tt_name} in {results_zip}")
                continue
            lc_bytes = z.read(lc_name)
            tt_bytes = z.read(tt_name)

        lc_text = lc_bytes.decode("utf-8").strip().splitlines()
        # header + 50 data rows = 51 lines
        if len(lc_text) != 51:
            errors.append(
                f"{lc_name}: expected 51 lines (header + 50 rows), got {len(lc_text)}"
            )
            continue
        last_row = lc_text[-1].split(",")
        if last_row[0] != "500000":
            errors.append(
                f"{lc_name}: last row's timestep is {last_row[0]!r}, expected '500000'"
            )
            continue

        run_id = canonical_run_id(algo, world, reward_mode, seed, no_drifts)
        if run_id in seen_run_ids:
            errors.append(f"DUPLICATE run_id {run_id} (from {models_zip})")
            continue
        seen_run_ids.add(run_id)

        runs.append({
            "run_id": run_id, "phase": phase, "algo": algo, "world": world,
            "reward_mode": reward_mode, "drift": drift, "no_drifts": no_drifts,
            "seed": seed, "models_zip": models_zip, "results_zip": results_zip,
            "old_name": old_name, "config_text": config_text,
            "model_bytes": model_bytes, "lc_bytes": lc_bytes, "tt_bytes": tt_bytes,
        })

    if verbose:
        print(f"Validated {len(runs)} models.zip against the mapping "
              f"({len(errors)} error(s)).")
    return runs, errors


def list_smoke_test_files():
    """List every raw file containing '_999_' (smoke tests), for the report."""
    hits = []
    for p in RAW_DIR.rglob("*.zip"):
        with zipfile.ZipFile(p) as z:
            for n in z.namelist():
                if "_999_" in n:
                    hits.append(f"{p.relative_to(RAW_DIR)} :: {n}")
    return hits


def build(runs, dry_run=False):
    """Step 2: write results_verified/<phase>/<run_id>/ + PROVENANCE.csv."""
    provenance_rows = []

    for run in runs:
        run_dir = OUT_DIR / run["phase"] / run["run_id"]
        if not dry_run:
            run_dir.mkdir(parents=True, exist_ok=True)

        files_to_write = {
            "config.txt": run["config_text"].encode("utf-8"),
            "learning_curve.csv": run["lc_bytes"],
            "training_time.txt": run["tt_bytes"],
            "model.zip": run["model_bytes"],
        }
        sources = {
            "config.txt": run["models_zip"],
            "learning_curve.csv": run["results_zip"],
            "training_time.txt": run["results_zip"],
            "model.zip": run["models_zip"],
        }

        # Original eval files, only for unambiguous zips
        for eval_zip_rel, owned in UNAMBIGUOUS_EVAL_ZIPS.items():
            if (run["world"], run["reward_mode"]) not in owned:
                continue
            if run["drift"] != "drifts":
                continue  # all unambiguous zips are hard+drifts only
            eval_zip_path = RAW_DIR / eval_zip_rel
            if not eval_zip_path.exists():
                continue
            with zipfile.ZipFile(eval_zip_path) as z:
                names = set(z.namelist())
                for suffix in EVAL_SUFFIXES:
                    fname = f"{run['old_name']}{suffix}"
                    if fname in names:
                        out_name = suffix.lstrip("_")
                        files_to_write[out_name] = z.read(fname)
                        sources[out_name] = eval_zip_path

        for fname, data in files_to_write.items():
            if not dry_run:
                (run_dir / fname).write_bytes(data)
            provenance_rows.append({
                "run_id": run["run_id"],
                "file": fname,
                "source_zip": str(sources[fname].relative_to(RAW_DIR)),
                "origin": "original-colab",
                "sha256": sha256_bytes(data),
            })

    if not dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / "PROVENANCE.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["run_id", "file", "source_zip", "origin", "sha256"])
            writer.writeheader()
            writer.writerows(provenance_rows)
        print(f"Wrote {len(provenance_rows)} provenance rows to {OUT_DIR / 'PROVENANCE.csv'}")

    return provenance_rows


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv

    runs, errors = validate_all()
    if errors:
        print(f"\n{len(errors)} ERROR(S) FOUND -- stopping before writing anything:\n")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print(f"\nAll {len(runs)} models.zip validated OK against the mapping.")
    by_phase = {}
    for r in runs:
        by_phase.setdefault(r["phase"], []).append(r["run_id"])
    for phase in ("phase1", "phase2", "phase3"):
        ids = sorted(by_phase.get(phase, []))
        print(f"  {phase}: {len(ids)} runs")
        for rid in ids:
            print(f"    - {rid}")

    smoke = list_smoke_test_files()
    print(f"\n{len(smoke)} smoke-test file(s) (containing '_999_') found and excluded:")
    for s in smoke:
        print(f"  - {s}")

    provenance_rows = build(runs, dry_run=dry_run)
    if dry_run:
        print(f"\n[DRY RUN] Would have written {len(provenance_rows)} files total. Nothing written.")
    else:
        print(f"\nWrote {len(runs)} run directories under {OUT_DIR}")
