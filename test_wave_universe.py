"""
Test Wave Universe Visibility

Verify that all 28 waves are recognized and available for rendering.
"""

import sys

# Test 1: Engine universe
print("=" * 80)
print("TEST 1: Wave Universe from Engine")
print("=" * 80)

from waves_engine import get_all_waves_universe, get_all_wave_ids

universe = get_all_waves_universe()
print(f"Total waves in universe: {universe['count']}")
print(f"Source: {universe['source']}")
print(f"Waves list length: {len(universe['waves'])}")
print(f"Wave IDs list length: {len(universe['wave_ids'])}")
print()

# Test 2: Data readiness
print("=" * 80)
print("TEST 2: Data Readiness Status (All Should Return is_ready=True)")
print("=" * 80)

from analytics_pipeline import compute_data_ready_status

all_ready = True
statuses = {}

for wave_id in get_all_wave_ids():
    status = compute_data_ready_status(wave_id)
    is_ready = status.get('is_ready', False)
    readiness_status = status.get('readiness_status', 'unknown')
    statuses[wave_id] = {'is_ready': is_ready, 'status': readiness_status}
    
    if not is_ready:
        all_ready = False
        print(f"✗ {wave_id}: is_ready={is_ready}, status={readiness_status}")

if all_ready:
    print("✓ All 28 waves return is_ready=True")
else:
    print("✗ Some waves return is_ready=False (this will block rendering)")

print()

# Test 3: Readiness distribution
print("=" * 80)
print("TEST 3: Readiness Distribution")
print("=" * 80)

status_counts = {}
for wave_id, info in statuses.items():
    status = info['status']
    status_counts[status] = status_counts.get(status, 0) + 1

for status, count in sorted(status_counts.items()):
    pct = count / len(statuses) * 100
    print(f"  {status:15} {count:2} waves ({pct:5.1f}%)")

print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total waves in system: {len(statuses)}")
print(f"All return is_ready=True: {all_ready}")
print()

if all_ready:
    print("✓ SUCCESS: All 28 waves will render in UI")
    sys.exit(0)
else:
    print("✗ FAILURE: Some waves may not render - check is_ready values")
    sys.exit(1)
