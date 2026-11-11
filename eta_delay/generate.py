"""Build shipment journeys from the definitions in definitions.py.

Schedule first, then realized durations, then propagation.

A stage starts at the later of its scheduled start and the previous stage's
actual finish. Finishing early does not pull the next stage forward. Delay
carries forward only when it is larger than the slack still left in the
schedule. The actual end of inland delivery is the ground-truth arrival.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from eta_delay.definitions import (
    AS_OF_STAGE,
    BUFFER_H,
    CONGESTION_SENSITIVITY_H,
    CUSTOMS_MISSING_PROBABILITY,
    DELAY_SIGN,
    DELAY_UNIT,
    DEPARTURE_ORIGIN,
    DEPARTURE_WINDOW_DAYS,
    MIN_DURATION_H,
    N_SHIPMENTS,
    NOISE_SIGMA_H,
    SEED,
    STAGES,
    ROUTES,
    Route,
    planned_duration_h,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
JOURNEYS_PATH = DATA_DIR / "journeys.csv"
SHIPMENTS_PATH = DATA_DIR / "shipments.csv"

JOURNEY_FIELDS = [
    "shipment_id",
    "route_id",
    "origin",
    "destination",
    "stage_order",
    "stage",
    "congestion_index",
    "planned_duration_h",
    "buffer_h",
    "scheduled_duration_h",
    "noise_h",
    "congestion_effect_h",
    "actual_duration_h",
    "duration_was_clipped",
    "scheduled_start_utc",
    "scheduled_end_utc",
    "truth_start_utc",
    "truth_end_utc",
    "observed_start_utc",
    "observed_end_utc",
    "is_observed",
    "carried_in_h",
    "carried_out_h",
    "generator_seed",
]

SHIPMENT_FIELDS = [
    "shipment_id",
    "route_id",
    "origin",
    "destination",
    "booked_departure_utc",
    "scheduled_arrival_utc",
    "actual_arrival_utc",
    "final_delay_h",
    "customs_observed",
    "congestion_port_of_loading",
    "congestion_port_of_discharge",
    "congestion_customs",
    "generator_seed",
    "delay_unit",
    "delay_sign",
    "as_of_stage",
]


@dataclass(frozen=True)
class StageRecord:
    shipment_id: str
    route_id: str
    origin: str
    destination: str
    stage_order: int
    stage: str
    congestion_index: float
    planned_duration_h: float
    buffer_h: float
    scheduled_duration_h: float
    noise_h: float
    congestion_effect_h: float
    actual_duration_h: float
    duration_was_clipped: bool
    scheduled_start_utc: datetime
    scheduled_end_utc: datetime
    truth_start_utc: datetime
    truth_end_utc: datetime
    observed_start_utc: datetime | None
    observed_end_utc: datetime | None
    is_observed: bool
    carried_in_h: float
    carried_out_h: float
    generator_seed: int


@dataclass(frozen=True)
class ShipmentRecord:
    shipment_id: str
    route_id: str
    origin: str
    destination: str
    booked_departure_utc: datetime
    scheduled_arrival_utc: datetime
    actual_arrival_utc: datetime
    final_delay_h: float
    customs_observed: bool
    congestion_port_of_loading: float
    congestion_port_of_discharge: float
    congestion_customs: float
    generator_seed: int
    delay_unit: str = DELAY_UNIT
    delay_sign: str = DELAY_SIGN
    as_of_stage: str = AS_OF_STAGE


def _iso(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def simulate_shipment(
    shipment_id: str,
    route: Route,
    booked_departure: datetime,
    congestion: dict[str, float],
    noise: dict[str, float],
    customs_observed: bool,
    generator_seed: int = SEED,
) -> tuple[list[StageRecord], ShipmentRecord]:
    """Apply the written mechanism to one shipment.

    congestion maps each stage to an index in [0, 1]. noise maps each stage
    to an hours offset, before the one-hour minimum duration is applied.
    """
    if booked_departure.tzinfo is None:
        raise ValueError("booked_departure must be timezone-aware UTC")

    stages: list[StageRecord] = []
    carried_in = 0.0
    cursor = booked_departure.astimezone(UTC)

    for order, stage in enumerate(STAGES, start=1):
        planned = planned_duration_h(route, stage)
        buffer = BUFFER_H[stage]
        scheduled_duration = planned + buffer
        effect = congestion[stage] * CONGESTION_SENSITIVITY_H[stage]
        raw_duration = planned + effect + noise[stage]
        actual_duration = max(MIN_DURATION_H, raw_duration)
        scheduled_start = cursor
        scheduled_end = scheduled_start + timedelta(hours=scheduled_duration)
        truth_start = scheduled_start + timedelta(hours=carried_in)
        truth_end = truth_start + timedelta(hours=actual_duration)
        carried_out = (truth_end - scheduled_end).total_seconds() / 3600.0
        observed = customs_observed or stage != "customs"

        stages.append(
            StageRecord(
                shipment_id=shipment_id,
                route_id=route.route_id,
                origin=route.origin,
                destination=route.destination,
                stage_order=order,
                stage=stage,
                congestion_index=float(congestion[stage]),
                planned_duration_h=planned,
                buffer_h=buffer,
                scheduled_duration_h=scheduled_duration,
                noise_h=float(noise[stage]),
                congestion_effect_h=float(effect),
                actual_duration_h=float(actual_duration),
                duration_was_clipped=actual_duration != raw_duration,
                scheduled_start_utc=scheduled_start,
                scheduled_end_utc=scheduled_end,
                truth_start_utc=truth_start,
                truth_end_utc=truth_end,
                observed_start_utc=truth_start if observed else None,
                observed_end_utc=truth_end if observed else None,
                is_observed=observed,
                carried_in_h=float(carried_in),
                carried_out_h=float(carried_out),
                generator_seed=generator_seed,
            )
        )
        carried_in = max(0.0, carried_out)
        cursor = scheduled_end

    last = stages[-1]
    shipment = ShipmentRecord(
        shipment_id=shipment_id,
        route_id=route.route_id,
        origin=route.origin,
        destination=route.destination,
        booked_departure_utc=booked_departure.astimezone(UTC),
        scheduled_arrival_utc=last.scheduled_end_utc,
        actual_arrival_utc=last.truth_end_utc,
        final_delay_h=last.carried_out_h,
        customs_observed=customs_observed,
        congestion_port_of_loading=float(congestion["port_of_loading"]),
        congestion_port_of_discharge=float(congestion["port_of_discharge"]),
        congestion_customs=float(congestion["customs"]),
        generator_seed=generator_seed,
    )
    return stages, shipment


def hand_example() -> tuple[list[StageRecord], ShipmentRecord]:
    """One fully determined journey used to audit the mechanism by hand.

    Route INNSA-AEJEA leaves on 2026-01-15 00:00 UTC. Discharge congestion
    is 1 and customs congestion is 0.5. Every noise term is 0. The worked
    arithmetic is in report/01_definitions_and_data.md and expects a final
    delay of 26 hours.
    """
    route = next(route for route in ROUTES if route.route_id == "INNSA-AEJEA")
    congestion = {stage: 0.0 for stage in STAGES}
    congestion["port_of_discharge"] = 1.0
    congestion["customs"] = 0.5
    noise = {stage: 0.0 for stage in STAGES}
    return simulate_shipment(
        shipment_id="EX-001",
        route=route,
        booked_departure=datetime(2026, 1, 15, tzinfo=UTC),
        congestion=congestion,
        noise=noise,
        customs_observed=True,
        generator_seed=SEED,
    )


def generate_dataset(
    n_shipments: int = N_SHIPMENTS,
    seed: int = SEED,
) -> tuple[list[StageRecord], list[ShipmentRecord]]:
    """Draw n_shipments journeys. The same seed always returns the same rows."""
    rng = np.random.default_rng(seed)
    journeys: list[StageRecord] = []
    shipments: list[ShipmentRecord] = []
    origin_date = datetime(*DEPARTURE_ORIGIN, tzinfo=UTC)

    for index in range(1, n_shipments + 1):
        route = ROUTES[int(rng.integers(0, len(ROUTES)))]
        day = int(rng.integers(0, DEPARTURE_WINDOW_DAYS))
        hour = int(rng.integers(0, 24))
        booked = origin_date + timedelta(days=day, hours=hour)
        congestion = {
            stage: float(rng.random()) if CONGESTION_SENSITIVITY_H[stage] else 0.0
            for stage in STAGES
        }
        noise = {
            stage: float(rng.normal(0.0, NOISE_SIGMA_H[stage])) for stage in STAGES
        }
        customs_observed = bool(rng.random() >= CUSTOMS_MISSING_PROBABILITY)
        stages, shipment = simulate_shipment(
            shipment_id=f"SHP-{index:03d}",
            route=route,
            booked_departure=booked,
            congestion=congestion,
            noise=noise,
            customs_observed=customs_observed,
            generator_seed=seed,
        )
        journeys.extend(stages)
        shipments.append(shipment)

    return journeys, shipments


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stage_row(stage: StageRecord) -> dict[str, object]:
    row = asdict(stage)
    for key in (
        "scheduled_start_utc",
        "scheduled_end_utc",
        "truth_start_utc",
        "truth_end_utc",
        "observed_start_utc",
        "observed_end_utc",
    ):
        row[key] = _iso(row[key])
    return row


def _shipment_row(shipment: ShipmentRecord) -> dict[str, object]:
    row = asdict(shipment)
    for key in (
        "booked_departure_utc",
        "scheduled_arrival_utc",
        "actual_arrival_utc",
    ):
        row[key] = _iso(row[key])
    return row


def write_dataset(
    journeys: list[StageRecord],
    shipments: list[ShipmentRecord],
    journeys_path: Path = JOURNEYS_PATH,
    shipments_path: Path = SHIPMENTS_PATH,
) -> None:
    _write_csv(journeys_path, JOURNEY_FIELDS, [_stage_row(stage) for stage in journeys])
    _write_csv(
        shipments_path,
        SHIPMENT_FIELDS,
        [_shipment_row(shipment) for shipment in shipments],
    )
