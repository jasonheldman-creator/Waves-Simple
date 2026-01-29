import os
import sys
import pandas as pd
from datetime import datetime

# ------------------------------------------------------------
# Repo path bootstrap
# ------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ------------------------------------------------------------
# Engine imports (authoritative)
# ------------------------------------------------------------

from waves_engine import (
    compute_history_nav,
    get_all_waves,
)

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "live_snapshot.csv")

# ------------------------------------------------------------
# Horizons (trading days)
# ------------------------------------------------------------

HORIZONS = {
    "1d": 1,
    "30d": 30,
    "60d": 60,
    "365d": 365,
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def compute_return(nav: pd.Series, days: int):
    if nav is None or len(nav) <= days:
        return None
    start = nav.iloc[-days - 1]
    end = nav.iloc[-1]
    if start <= 0:
        return None
    return float(end / start - 1.0)


# ------------------------------------------------------------
# Main snapshot builder
# ------------------------------------------------------------

def generate_live_snapshot():
    rows = []
    asof = datetime.utcnow().date().isoformat()

    waves = get_all_waves()
    if not waves:
        raise RuntimeError("No waves returned by get_all_waves()")

    for wave_name in waves:
        try:
            df = compute_history_nav(
                wave_name=wave_name,
                mode="Standard",
                days=400,          # buffer > 365
                include_diagnostics=True
            )

            if df is None or df.empty:
                continue

            wave_nav = df["wave_nav"]
            bm_nav = df["bm_nav"]

            row = {
                "wave_name": wave_name,
                "asof": asof,
            }

            # ------------------------------------------------
            # Returns + Alpha
            # ------------------------------------------------

            for label, days in HORIZONS.items():
                w_ret = compute_return(wave_nav, days)
                b_ret = compute_return(bm_nav, days)

                row[f"return_{label}"] = w_ret
                row[f"alpha_{label}"] = (
                    w_ret - b_ret if w_ret is not None and b_ret is not None else None
                )

            # ------------------------------------------------
            # Attribution placeholders (STRUCTURAL CONTRACT)
            # ------------------------------------------------
            # These must exist or the UI WILL BLANK.
            # They are filled once true attribution math is finalized.

            for label in HORIZONS.keys():
                row[f"alpha_selection_{label}"] = row.get(f"alpha_{label}")
                row[f"alpha_momentum_{label}"] = 0.0
                row[f"alpha_volatility_{label}"] = 0.0
                row[f"alpha_beta_{label}"] = 0.0
                row[f"alpha_allocation_{label}"] = 0.0
                row[f"alpha_residual_{label}"] = row.get(f"alpha_{label}")

            rows.append(row)

        except Exception as e:
            print(f"❌ Snapshot error for {wave_name}: {e}")
            continue

    if not rows:
        raise RuntimeError("Snapshot generation failed for all waves")

    df_out = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Live snapshot written: {OUTPUT_PATH}")
    print(f"   Waves: {len(df_out)}")
    print(f"   Columns: {len(df_out.columns)}")


# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------

if __name__ == "__main__":
    generate_live_snapshot()