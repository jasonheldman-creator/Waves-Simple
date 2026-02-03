from pathlib import Path
import pandas as pd
from datetime import date

# -------------------------------------------------------------------
# CANONICAL SNAPSHOT GENERATOR
# -------------------------------------------------------------------
# This file is the ONE AND ONLY place where canonical snapshot rows
# are generated. Every Wave gets exactly one row.
# -------------------------------------------------------------------

CANONICAL_WAVES = [
    ("ai_cloud_megacap_wave", "AI & Cloud MegaCap Wave", "Equity"),
    ("clean_transit_infrastructure_wave", "Clean Transit-Infrastructure Wave", "Equity"),
    ("crypto_ai_growth_wave", "Crypto AI Growth Wave", "Crypto"),
    ("crypto_broad_growth_wave", "Crypto Broad Growth Wave", "Crypto"),
    ("crypto_defi_growth_wave", "Crypto DeFi Growth Wave", "Crypto"),
    ("crypto_income_wave", "Crypto Income Wave", "Crypto"),
    ("crypto_l1_growth_wave", "Crypto L1 Growth Wave", "Crypto"),
    ("crypto_l2_growth_wave", "Crypto L2 Growth Wave", "Crypto"),
    ("demas_fund_wave", "Demas Fund Wave", "Equity"),
    ("ev_infrastructure_wave", "EV & Infrastructure Wave", "Equity"),
    ("future_energy_ev_wave", "Future Energy & EV Wave", "Equity"),
    ("future_power_energy_wave", "Future Power & Energy Wave", "Equity"),
    ("gold_wave", "Gold Wave", "Commodity"),
    ("income_wave", "Income Wave", "Fixed Income"),
    ("infinity_multi_asset_growth_wave", "Infinity Multi-Asset Growth Wave", "Multi-Asset"),
    ("next_gen_compute_semis_wave", "Next-Gen Compute & Semis Wave", "Equity"),
    ("quantum_computing_wave", "Quantum Computing Wave", "Equity"),
    ("russell_3000_wave", "Russell 3000 Wave", "Equity"),
    ("sp500_wave", "S&P 500 Wave", "Equity"),
    ("small_cap_growth_wave", "Small Cap Growth Wave", "Equity"),
    ("small_to_mid_cap_growth_wave", "Small-to-Mid Cap Growth Wave", "Equity"),
    ("smartsafe_tax_free_money_market_wave", "SmartSafe Tax-Free MM Wave", "Cash"),
    ("smartsafe_treasury_cash_wave", "SmartSafe Treasury Cash Wave", "Cash"),
    ("us_megacap_core_wave", "US MegaCap Core Wave", "Equity"),
    ("us_mid_small_growth_semis_wave", "US Mid/Small Growth Semis Wave", "Equity"),
    ("us_small_cap_disruptors_wave", "US Small-Cap Disruptors Wave", "Equity"),
    ("vector_muni_ladder_wave", "Vector Muni Ladder Wave", "Fixed Income"),
    ("vector_treasury_ladder_wave", "Vector Treasury Ladder Wave", "Fixed Income"),
]

SNAPSHOT_COLUMNS = [
    "Wave_ID",
    "Wave_Name",
    "Asset_Class",
    "Mode",
    "Snapshot_Date",
    "Return_1D",
    "Return_30D",
    "Return_60D",
    "Return_365D",
    "Alpha_1D",
    "Alpha_30D",
    "Alpha_60D",
    "Alpha_365D",
    "Benchmark_Return_1D",
    "Benchmark_Return_30D",
    "Benchmark_Return_60D",
    "Benchmark_Return_365D",
    "VIX_Regime",
    "Exposure",
    "CashPercent",
]

def build_canonical_snapshot(
    output_path: Path = Path("data/canonical_snapshot.csv"),
) -> pd.DataFrame:
    today = date.today().isoformat()
    rows = []

    for wave_id, wave_name, asset_class in CANONICAL_WAVES:
        rows.append([
            wave_id,
            wave_name,
            asset_class,
            "Standard",
            today,
            None, None, None, None,
            None, None, None, None,
            None, None, None, None,
            "UNKNOWN",
            1.0,
            0.0,
        ])

    df = pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, header=False)

    print(f"✓ Canonical snapshot written: {output_path}")
    print(f"✓ Rows: {len(df)} | Columns: {len(df.columns)}")

    return df


if __name__ == "__main__":
    build_canonical_snapshot()