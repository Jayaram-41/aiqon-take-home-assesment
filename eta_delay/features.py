"""Information available at the end of port of discharge.

Completed stages may contribute their observed durations. Customs and inland
delivery have not happened yet, so their truth timestamps and true durations
are not features. Congestion indexes are treated as known at this moment,
including the customs index for the stage about to start.

The training label for a stage duration is the true duration, including
customs clearances whose milestone was never recorded. That label is not a
feature.
"""

from __future__ import annotations

from dataclasses import dataclass

from eta_delay.definitions import AS_OF_STAGE, STAGES
from eta_delay.generate import ShipmentRecord, StageRecord
from eta_delay.propagate import StageInput, replay

AS_OF_ORDER = STAGES.index(AS_OF_STAGE) + 1

DURATION_FEATURE_FIELDS = (
    "stage",
    "planned_duration_h",
    "congestion_index",
)


@dataclass(frozen=True)
class DurationExample:
    shipment_id: str
    stage: str
    planned_duration_h: float
    congestion_index: float
    label_duration_h: float

    def features(self) -> dict[str, float | str]:
        return {
            "stage": self.stage,
            "planned_duration_h": self.planned_duration_h,
            "congestion_index": self.congestion_index,
        }


@dataclass(frozen=True)
class StageView:
    stage: str
    planned_duration_h: float
    buffer_h: float
    scheduled_duration_h: float
    congestion_index: float
    observed_duration_h: float | None


@dataclass(frozen=True)
class AsOfState:
    """One shipment as seen when port of discharge has just finished."""

    shipment_id: str
    route_id: str
    origin: str
    destination: str
    scheduled_arrival_utc: object
    carried_out_at_as_of_h: float
    completed: tuple[StageView, ...]
    remaining: tuple[StageView, ...]
    label_final_delay_h: float

    def congestion(self, stage: str) -> float:
        for item in (*self.completed, *self.remaining):
            if item.stage == stage:
                return item.congestion_index
        raise KeyError(stage)


def duration_examples(journeys: list[StageRecord]) -> list[DurationExample]:
    """One training row per stage. The label is kept off the feature dict."""
    return [
        DurationExample(
            shipment_id=stage.shipment_id,
            stage=stage.stage,
            planned_duration_h=stage.planned_duration_h,
            congestion_index=stage.congestion_index,
            label_duration_h=stage.actual_duration_h,
        )
        for stage in journeys
    ]


def _observed_duration_h(stage: StageRecord) -> float:
    if not stage.is_observed or stage.observed_start_utc is None or stage.observed_end_utc is None:
        raise ValueError(f"{stage.shipment_id} {stage.stage} has no observed timestamps")
    return (stage.observed_end_utc - stage.observed_start_utc).total_seconds() / 3600.0


def _view(stage: StageRecord, observed_duration_h: float | None) -> StageView:
    return StageView(
        stage=stage.stage,
        planned_duration_h=stage.planned_duration_h,
        buffer_h=stage.buffer_h,
        scheduled_duration_h=stage.scheduled_duration_h,
        congestion_index=stage.congestion_index,
        observed_duration_h=observed_duration_h,
    )


def build_as_of_states(
    journeys: list[StageRecord],
    shipments: list[ShipmentRecord],
) -> list[AsOfState]:
    """Build the decision-time view for every shipment."""
    by_shipment: dict[str, list[StageRecord]] = {}
    for stage in journeys:
        by_shipment.setdefault(stage.shipment_id, []).append(stage)

    states: list[AsOfState] = []
    for shipment in shipments:
        stages = sorted(by_shipment[shipment.shipment_id], key=lambda stage: stage.stage_order)
        if [stage.stage for stage in stages] != list(STAGES):
            raise ValueError(f"{shipment.shipment_id} does not have the six defined stages")

        views: list[StageView] = []
        for stage in stages:
            if stage.stage_order <= AS_OF_ORDER:
                views.append(_view(stage, _observed_duration_h(stage)))
            else:
                views.append(_view(stage, None))

        completed = tuple(view for view in views if view.observed_duration_h is not None)
        remaining = tuple(view for view in views if view.observed_duration_h is None)
        carried_out = replay(
            [
                StageInput(
                    stage=view.stage,
                    scheduled_duration_h=view.scheduled_duration_h,
                    duration_h=float(view.observed_duration_h),
                    source="observed",
                )
                for view in completed
            ]
        ).predicted_delay_h
        states.append(
            AsOfState(
                shipment_id=shipment.shipment_id,
                route_id=shipment.route_id,
                origin=shipment.origin,
                destination=shipment.destination,
                scheduled_arrival_utc=shipment.scheduled_arrival_utc,
                carried_out_at_as_of_h=carried_out,
                completed=completed,
                remaining=remaining,
                label_final_delay_h=shipment.final_delay_h,
            )
        )
    return states
