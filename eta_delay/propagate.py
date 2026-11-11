"""Turn stage durations into a final delay, and say which hours survive.

This is the same rule as the generator. It is not a second statistical model.

A stage contributes to the final delay only when the shipment never gets
back onto the schedule after that stage. Earliness is discarded, because a
stage cannot start early. Once the journey is back on time, earlier stages
no longer affect the consignee.

On the worked example EX-001 the excesses are -2, -8, -12, +30, +2, -6.
Ocean transit finishes 12 hours inside its schedule, which resets the
chain. Only discharge, customs, and inland remain: 30 + 2 - 6 = 26, the
final delay.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageInput:
    stage: str
    scheduled_duration_h: float
    duration_h: float
    source: str


@dataclass(frozen=True)
class StageResult:
    stage: str
    source: str
    duration_h: float
    scheduled_duration_h: float
    excess_h: float
    carried_in_h: float
    carried_out_h: float
    contribution_h: float


@dataclass(frozen=True)
class Replay:
    stages: tuple[StageResult, ...]
    predicted_delay_h: float
    reset_stage: str | None

    @property
    def contribution_sum_h(self) -> float:
        return sum(stage.contribution_h for stage in self.stages)


def replay(inputs: list[StageInput] | tuple[StageInput, ...]) -> Replay:
    """Propagate durations in order and attribute the final delay."""
    if not inputs:
        raise ValueError("replay needs at least one stage")

    carried_in = 0.0
    excesses: list[float] = []
    carried_out: list[float] = []
    for item in inputs:
        excess = item.duration_h - item.scheduled_duration_h
        leaving = carried_in + excess
        excesses.append(excess)
        carried_out.append(leaving)
        carried_in = max(0.0, leaving)

    contributions, reset_stage = _attribute(inputs, excesses, carried_out)
    stages = tuple(
        StageResult(
            stage=item.stage,
            source=item.source,
            duration_h=item.duration_h,
            scheduled_duration_h=item.scheduled_duration_h,
            excess_h=excesses[index],
            carried_in_h=0.0 if index == 0 else max(0.0, carried_out[index - 1]),
            carried_out_h=carried_out[index],
            contribution_h=contributions[index],
        )
        for index, item in enumerate(inputs)
    )
    predicted = carried_out[-1]
    if abs(sum(contributions) - predicted) > 1e-6:
        raise RuntimeError("stage contributions do not add up to the predicted delay")
    return Replay(stages=stages, predicted_delay_h=predicted, reset_stage=reset_stage)


def _attribute(
    inputs: list[StageInput] | tuple[StageInput, ...],
    excesses: list[float],
    carried_out: list[float],
) -> tuple[list[float], str | None]:
    """Hours of the final delay that each stage still accounts for."""
    last_reset = -1
    for index in range(len(inputs) - 1):
        if carried_out[index] <= 1e-9:
            last_reset = index
    start = last_reset + 1
    contributions = [0.0] * len(inputs)
    for index in range(start, len(inputs)):
        contributions[index] = excesses[index]
    reset_stage = None if last_reset < 0 else inputs[last_reset].stage
    return contributions, reset_stage
