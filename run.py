"""Pipeline orchestrator: fetch -> build -> backtest -> evaluate -> plots.

    python run.py            # run every stage in order
    python run.py backtest   # run one stage (or several: `run.py build plots`)

Each stage is a thin call into the corresponding src/ module so this file
stays a table of contents, not a place where logic hides.
"""
from __future__ import annotations

import argparse
import sys


def stage_fetch() -> None:
    from src.config import ECONDATA_DATASET_ID, SARB_SERIES
    from src.data.fred_client import fetch_fred_series
    from src.data.local_csv import convert_local_csvs

    convert_local_csvs()
    fetch_fred_series()

    # SARB fetch only once discovery has been done and config filled in;
    # until then we explain what to do instead of crashing mid-pipeline.
    if ECONDATA_DATASET_ID and all(SARB_SERIES.values()):
        from src.data.econdata_client import fetch_sarb_series

        fetch_sarb_series(ECONDATA_DATASET_ID, SARB_SERIES)
    else:
        print(
            "\n[fetch] SARB series not configured yet.\n"
            "  1. copy .env.example to .env and add ECONDATA_CREDENTIALS\n"
            "  2. python -m src.data.econdata_client <DATASET_ID> [filter]\n"
            "  3. paste ECONDATA_DATASET_ID and the SARB_SERIES keys into src/config.py\n"
            "  4. re-run: python run.py fetch\n"
        )
        sys.exit(1)


def stage_build() -> None:
    from src.data.build_dataset import main

    main()


def stage_backtest() -> None:
    from src.backtest import main

    main()


def stage_evaluate() -> None:
    from src.evaluate import main

    main()


def stage_plots() -> None:
    from src.plots import main

    main()


STAGES = {
    "fetch": stage_fetch,
    "build": stage_build,
    "backtest": stage_backtest,
    "evaluate": stage_evaluate,
    "plots": stage_plots,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="*", choices=[*STAGES, []],
                        help="stages to run (default: all, in order)")
    args = parser.parse_args()
    for name in args.stages or list(STAGES):
        print(f"\n===== stage: {name} =====")
        STAGES[name]()


if __name__ == "__main__":
    main()
