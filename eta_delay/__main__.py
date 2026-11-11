"""Generate data/journeys.csv and data/shipments.csv, then run the checks."""

from __future__ import annotations

import sys

from eta_delay.checks import check_dataset, format_report
from eta_delay.generate import generate_dataset, write_dataset


def main() -> int:
    journeys, shipments = generate_dataset()
    write_dataset(journeys, shipments)
    report = check_dataset(journeys, shipments)
    print(format_report(report))
    print("wrote data/journeys.csv and data/shipments.csv")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
