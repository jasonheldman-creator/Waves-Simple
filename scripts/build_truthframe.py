import json
from datetime import datetime, timezone

from helpers.price_book import get_price_book
from helpers.universal_universe import get_wave_universe_all
from helpers.wave_data import get_wave_data_filtered


def build_truthframe(days: int = 60):
    price_book = get_price_book()

    if price_book is None or price_book.empty:
        raise RuntimeError("PRICE_BOOK is empty — cannot build TruthFrame")

    truth = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "price_book_rows": len(price_book),
            "price_book_cols": len(price_book.columns),
            "lookback_days": days,
        },
        "waves": {},
    }

    waves = get_wave_universe_all()

    for wave in waves:
        try:
            df = get_wave_data_filtered(wave, days)

            if df is None or df.empty:
                raise ValueError("No wave data")

            df = df.copy()
            df["alpha"] = df["portfolio_return"] - df["benchmark_return"]

            total_alpha = float(df["alpha"].sum())
            exposure = float(df["exposure"].mean()) if "exposure" in df.columns else 1.0

            truth["waves"][wave] = {
                "alpha": {
                    "total": total_alpha,
                    "selection": total_alpha * exposure,
                    "overlay": total_alpha * (1 - exposure) * 0.7,
                    "cash": total_alpha * (1 - exposure) * 0.3,
                },
                "health": {"status": "OK"},
                "learning": {},
            }

        except Exception as e:
            truth["waves"][wave] = {
                "alpha": {
                    "total": 0.0,
                    "selection": 0.0,
                    "overlay": 0.0,
                    "cash": 0.0,
                },
                "health": {"status": "DEGRADED", "error": str(e)},
                "learning": {},
            }

    return truth


if __name__ == "__main__":
    truthframe = build_truthframe()

    with open("data/truthframe.json", "w") as f:
        json.dump(truthframe, f, indent=2)

    print("✅ TruthFrame written to data/truthframe.json")