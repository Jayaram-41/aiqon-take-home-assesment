"""Sanity checks for a generated dataset.

These checks do not score a model. They confirm the file matches the
mechanism: slack sometimes absorbs delay, delay is not a constant, and
some customs timestamps are unobserved.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from eta_delay.definitions import N_SHIPMENTS, STAGES
from eta_delay.generate import ShipmentRecord, StageRecord


@dataclass(frozen=True)
class CheckReport:
    n_shipments: int
    n_stage_rows: int
    n_on_time: int
    n_late: int
    delay_min_h: float
    delay_mean_h: float
    delay_median_h: float
    delay_max_h: float
    n_customs_missing: int
    routes: dict[str, int]
    failures: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.failures


def _hours_between(end, start) -> float:
    return (end - start).total_seconds() / 3600.0


def propagation_errors(stages: list[StageRecord]) -> list[str]:
    """Return identity violations for one shipment's stages, in order."""
    ordered = sorted(stages, key=lambda stage: stage.stage_order)
    errors: list[str] = []
    if [stage.stage for stage in ordered] != list(STAGES):
        errors.append("stage list is not the defined six stages")
        return errors

    previous_out = 0.0
    previous_scheduled_end = None
    for stage in ordered:
        expected_in = 0.0 if previous_scheduled_end is None else max(0.0, previous_out)
        if abs(stage.carried_in_h - expected_in) > 1e-6:
            errors.append(f"{stage.stage} carried_in_h is not the previous lateness")
        if previous_scheduled_end is not None and stage.scheduled_start_utc != previous_scheduled_end:
            errors.append(f"{stage.stage} does not start when the previous schedule ends")
        if stage.truth_start_utc < stage.scheduled_start_utc:
            errors.append(f"{stage.stage} starts before its scheduled start")
        duration_h = _hours_between(stage.truth_end_utc, stage.truth_start_utc)
        if abs(duration_h - stage.actual_duration_h) > 1e-6:
            errors.append(f"{stage.stage} actual duration does not match its timestamps")
        scheduled_h = _hours_between(stage.scheduled_end_utc, stage.scheduled_start_utc)
        if abs(scheduled_h - stage.scheduled_duration_h) > 1e-6:
            errors.append(f"{stage.stage} scheduled duration does not match its timestamps")
        carried_out = _hours_between(stage.truth_end_utc, stage.scheduled_end_utc)
        if abs(carried_out - stage.carried_out_h) > 1e-6:
            errors.append(f"{stage.stage} carried_out_h is not truth end minus scheduled end")
        if stage.stage == "customs" and not stage.is_observed:
            if stage.observed_start_utc is not None or stage.observed_end_utc is not None:
                errors.append("unobserved customs still has observed timestamps")
        elif stage.observed_start_utc != stage.truth_start_utc:
            errors.append(f"{stage.stage} observed start differs from the truth")
        previous_out = stage.carried_out_h
        previous_scheduled_end = stage.scheduled_end_utc
    return errors


def check_dataset(
    journeys: list[StageRecord],
    shipments: list[ShipmentRecord],
    n_shipments: int = N_SHIPMENTS,
) -> CheckReport:
    failures: list[str] = []
    if len(shipments) != n_shipments:
        failures.append(f"expected {n_shipments} shipments, found {len(shipments)}")
    if len(journeys) != n_shipments * len(STAGES):
        failures.append(
            f"expected {n_shipments * len(STAGES)} stage rows, found {len(journeys)}"
        )

    by_shipment: dict[str, list[StageRecord]] = {}
    for stage in journeys:
        by_shipment.setdefault(stage.shipment_id, []).append(stage)

    for shipment in shipments:
        stages = by_shipment.get(shipment.shipment_id, [])
        for message in propagation_errors(stages):
            failures.append(f"{shipment.shipment_id}: {message}")
        label_h = _hours_between(shipment.actual_arrival_utc, shipment.scheduled_arrival_utc)
        if abs(label_h - shipment.final_delay_h) > 1e-6:
            failures.append(f"{shipment.shipment_id}: final delay does not match the two arrivals")
        if stages and abs(stages[-1].carried_out_h - shipment.final_delay_h) > 1e-6:
            failures.append(f"{shipment.shipment_id}: final delay is not the inland carried_out")
        customs = next(stage for stage in stages if stage.stage == "customs")
        if customs.is_observed != shipment.customs_observed:
            failures.append(f"{shipment.shipment_id}: customs flag disagrees with the stage row")

    delays = np.array([shipment.final_delay_h for shipment in shipments], dtype=float)
    on_time = int(np.sum(delays <= 0))
    late = int(np.sum(delays > 0))
    missing = sum(1 for shipment in shipments if not shipment.customs_observed)
    routes = dict(Counter(shipment.route_id for shipment in shipments))

    if len(delays):
        if on_time < 0.15 * len(delays):
            failures.append(
                f"only {on_time} shipments are on time; slack should absorb some delay"
            )
        if late < 0.15 * len(delays):
            failures.append(f"only {late} shipments are late; delay should reach the consignee")
        if float(np.std(delays)) <= 1.0:
            failures.append("final delay is nearly constant")
    if missing < 0.10 * len(shipments):
        failures.append(f"only {missing} customs milestones are missing")
    if len(routes) < 4:
        failures.append(f"expected all four routes, found {sorted(routes)}")

    return CheckReport(
        n_shipments=len(shipments),
        n_stage_rows=len(journeys),
        n_on_time=on_time,
        n_late=late,
        delay_min_h=float(np.min(delays)) if len(delays) else 0.0,
        delay_mean_h=float(np.mean(delays)) if len(delays) else 0.0,
        delay_median_h=float(np.median(delays)) if len(delays) else 0.0,
        delay_max_h=float(np.max(delays)) if len(delays) else 0.0,
        n_customs_missing=missing,
        routes=routes,
        failures=tuple(failures),
    )


def format_report(report: CheckReport) -> str:
    lines = [
        f"shipments: {report.n_shipments}",
        f"stage rows: {report.n_stage_rows}",
        f"on time (delay <= 0): {report.n_on_time}",
        f"late (delay > 0): {report.n_late}",
        (
            "final delay hours: "
            f"min {report.delay_min_h:.1f}, median {report.delay_median_h:.1f}, "
            f"mean {report.delay_mean_h:.1f}, max {report.delay_max_h:.1f}"
        ),
        f"customs milestones missing: {report.n_customs_missing}",
        "routes: " + ", ".join(f"{key} {value}" for key, value in sorted(report.routes.items())),
    ]
    if report.failures:
        lines.append("FAILED")
        lines.extend(f"- {failure}" for failure in report.failures[:12])
    else:
        lines.append("checks passed")
    return "\n".join(lines)
