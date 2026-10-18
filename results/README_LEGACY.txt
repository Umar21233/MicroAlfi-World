This folder (results/) is UNVERIFIED and SUPERSEDED.

Its run-name/CSV-naming convention ({algo}_{world}_{reward_mode}_{seed},
with no drift-condition component) means a with-drifts and a no-drifts run
of the same algo/world/reward_mode/seed silently overwrite each other's
files. docs/AUDIT.md (finding #12) documents this. Several files in this folder
are therefore known to be contaminated: the most concrete example is
dqn_hard_multi_objective_42/43/44's eval numbers here (mean ~4958) not
matching the officially reported Phase 1 figure (5221) or the verified
figure rebuilt from the original Colab training outputs (5220.89).

The trustworthy, provenance-tracked dataset is results_verified/, rebuilt
directly from the raw "Colab training results" downloads (read-only,
SHA-256-tracked via data_manifest.json) with a collision-free 5-part
run-id scheme ({algo}_{world}_{reward_mode}_{drift}_{seed}). See
docs/VERIFICATION.md for the full rebuild process, and README.md for the
folder structure and regeneration commands.

This folder is kept, unmodified, for historical reference only. Do not
delete it (per the audit's hard rules) and do not treat any number in it
as authoritative without cross-checking results_verified/.
