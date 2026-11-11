"""Targets, stages, and the rules that generate delay.

These constants are the contract for the dataset. Change them only by
editing this module, then regenerating the CSV files.
"""

from __future__ import annotations

from dataclasses import dataclass

# final_delay_hours = actual final arrival minus scheduled final arrival.
# The unit is hours. Positive means the shipment is late. Both instants are UTC.
DELAY_UNIT = "hours"
DELAY_SIGN = "positive_means_late"

# A later model may use actuals only through this stage. Customs and inland
# delivery are still in the future at that moment.
AS_OF_STAGE = "port_of_discharge"

STAGES: tuple[str, ...] = (
    "origin_departure",
    "port_of_loading",
    "ocean_transit",
    "port_of_discharge",
    "customs",
    "inland_delivery",
)

# Hours of work the schedule expects, before any buffer.
ORIGIN_PLANNED_H = 8.0
CUSTOMS_PLANNED_H = 20.0

# Extra hours the schedule allows beyond planned work. This is the slack
# that can absorb upstream delay. A stage still cannot start early.
BUFFER_H: dict[str, float] = {
    "origin_departure": 2.0,
    "port_of_loading": 8.0,
    "ocean_transit": 12.0,
    "port_of_discharge": 10.0,
    "customs": 6.0,
    "inland_delivery": 6.0,
}

# Added hours when the congestion index is 1. Ocean, origin, and inland
# delivery are unaffected in this dataset. Weather is not simulated.
CONGESTION_SENSITIVITY_H: dict[str, float] = {
    "origin_departure": 0.0,
    "port_of_loading": 20.0,
    "ocean_transit": 0.0,
    "port_of_discharge": 40.0,
    "customs": 16.0,
    "inland_delivery": 0.0,
}

# Standard deviation, in hours, of the noise added to each stage duration.
NOISE_SIGMA_H: dict[str, float] = {
    "origin_departure": 2.0,
    "port_of_loading": 5.0,
    "ocean_transit": 6.0,
    "port_of_discharge": 6.0,
    "customs": 4.0,
    "inland_delivery": 4.0,
}

MIN_DURATION_H = 1.0

N_SHIPMENTS = 200
SEED = 20261111
CUSTOMS_MISSING_PROBABILITY = 0.18
DEPARTURE_WINDOW_DAYS = 90
DEPARTURE_ORIGIN = (2026, 1, 1)


@dataclass(frozen=True)
class Route:
    route_id: str
    origin: str
    destination: str
    port_of_loading_h: float
    ocean_h: float
    port_of_discharge_h: float
    inland_h: float


# UN/LOCODE-style port codes. Planned hours differ by route so the same
# congestion index does not imply the same dwell everywhere.
ROUTES: tuple[Route, ...] = (
    Route("CNSHA-NLRTM", "CNSHA", "NLRTM", 30.0, 720.0, 40.0, 36.0),
    Route("SGSIN-USLAX", "SGSIN", "USLAX", 22.0, 504.0, 48.0, 28.0),
    Route("INNSA-AEJEA", "INNSA", "AEJEA", 28.0, 120.0, 30.0, 30.0),
    Route("KRPUS-DEHAM", "KRPUS", "DEHAM", 26.0, 768.0, 42.0, 40.0),
)

ROUTES_BY_ID = {route.route_id: route for route in ROUTES}


def planned_duration_h(route: Route, stage: str) -> float:
    """Planned work time for one stage, in hours, with no buffer and no noise."""
    if stage == "origin_departure":
        return ORIGIN_PLANNED_H
    if stage == "port_of_loading":
        return route.port_of_loading_h
    if stage == "ocean_transit":
        return route.ocean_h
    if stage == "port_of_discharge":
        return route.port_of_discharge_h
    if stage == "customs":
        return CUSTOMS_PLANNED_H
    if stage == "inland_delivery":
        return route.inland_h
    raise KeyError(stage)
