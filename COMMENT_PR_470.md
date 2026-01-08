MESSAGE TO COPILOT (PR #470 ONLY):

1. analytics_truth.py / generate_live_snapshot_csv() — fix root cause, not more prints

• Remove/disable all earlier wave_id uniqueness assertions. There must be exactly one validation block and it must be immediately before the return of the final snapshot DataFrame.
• Derive expected_wave_ids dynamically from wave_weights.csv (source of truth). Do not hardcode 28. Set expected_count = len(expected_wave_ids).
• Normalize wave_id column before validation (this is the bug):
• Preserve a raw copy: df['wave_id_raw'] = df['wave_id']
• Convert to string safely, strip whitespace:
• treat None, NaN, pd.NA, "", "   " as invalid
• apply .astype("string") then .str.strip()
• replace blanks with pd.NA
• Guarantee wave_id is never null:
• If registry conversion returns None/NA for any row, fill with a deterministic slug built from display_name (e.g., lowercase, replace non-alphanum with _, collapse repeats, trim _). This must be deterministic and stable.
• After fallback fill, re-strip again.
• Validate once, using NA-aware metrics:
• nunique_dropna_true = df['wave_id'].nunique(dropna=True)
• nunique_dropna_false = df['wave_id'].nunique(dropna=False)
• isna_sum = df['wave_id'].isna().sum()
• blank_sum = (df['wave_id'].astype("string").str.strip() == "").sum()
• duplicates: dupes = df['wave_id'].value_counts(dropna=False); dupes = dupes[dupes > 1]
• expected/actual set diffs:
• expected_set = set(expected_wave_ids)
• actual_set = set(df['wave_id'].dropna().tolist())
• missing = sorted(expected_set - actual_set)
• unexpected = sorted(actual_set - expected_set)
• If ANY failure, raise ValueError with a multi-line message that prints:
• expected_count
• nunique(dropna=True) and nunique(dropna=False)
• isna_sum and blank_sum
• duplicates list + counts
• missing + unexpected IDs
• offending rows: for each bad row (NA/blank OR in duplicates OR unexpected OR mapped via fallback), print: display_name, wave_id_raw, normalized wave_id
• Success condition:
• isna_sum == 0
• blank_sum == 0
• len(dupes) == 0
• missing == []
• unexpected == []
• nunique(dropna=False) == expected_count

2. scripts/rebuild_snapshot.py

• Ensure generate_live_snapshot_csv() is called exactly once.
• Wrap in try/except:
• On exception: print the exception message (so the diagnostics appear in Actions logs) and sys.exit(1)
• On success: sys.exit(0)

3. .github/workflows/rebuild_snapshot.yml

• Confirm the rebuild script is invoked once (no duplicate steps calling rebuild_snapshot.py).
• Do not add jobs or new workflows.

Definition of done: Manually running “Rebuild Snapshot” on main succeeds (green). No more “Expected 28 unique wave_ids” failures.