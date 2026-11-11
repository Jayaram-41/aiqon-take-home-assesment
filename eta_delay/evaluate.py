"""Score final delay at the end of port of discharge.

The split is by shipment. Five folds, seed 20261111. Each shipment is
predicted once, from a model that did not train on it. The headline numbers
are mean absolute error in hours and signed bias. Bias is prediction minus
actual, so a positive bias means the predicted arrival is too late.

The explained shipment is the fold-0 holdout whose absolute error is closest
to that fold's median absolute error. It is a typical miss, not a best case.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from eta_delay.definitions import SEED
from eta_delay.features import AsOfState, build_as_of_states, duration_examples
from eta_delay.generate import DATA_DIR, ShipmentRecord, StageRecord, generate_dataset
from eta_delay.propagate import Replay
from eta_delay.stage_model import DurationModel, FlatDelayModel, predict_structured

N_FOLDS = 5
REPORT_PATH = Path(__file__).resolve().parents[1] / "report" / "02_model_results.md"
OOF_PATH = DATA_DIR / "oof_predictions.csv"


@dataclass(frozen=True)
class ShipPrediction:
    shipment_id: str
    route_id: str
    fold: int
    actual_delay_h: float
    schedule_h: float
    mean_h: float
    structured_h: float
    flat_h: float
    customs_duration_error_h: float
    inland_duration_error_h: float


@dataclass(frozen=True)
class ExplainedShipment:
    state: AsOfState
    replay: Replay
    fold: int
    predicted_arrival_utc: datetime


@dataclass(frozen=True)
class Evaluation:
    predictions: tuple[ShipPrediction, ...]
    explained: ExplainedShipment

    def column(self, name: str) -> np.ndarray:
        return np.array([getattr(row, name) for row in self.predictions], dtype=float)


def _folds(shipment_ids: list[str], n_folds: int, seed: int) -> list[list[str]]:
    order = np.array(shipment_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    return [list(chunk) for chunk in np.array_split(order, n_folds)]


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(predicted - actual))


def evaluate(
    journeys: list[StageRecord],
    shipments: list[ShipmentRecord],
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> Evaluation:
    states = build_as_of_states(journeys, shipments)
    examples = duration_examples(journeys)
    by_id = {state.shipment_id: state for state in states}
    truth_duration = {
        (stage.shipment_id, stage.stage): stage.actual_duration_h for stage in journeys
    }
    folds = _folds([state.shipment_id for state in states], n_folds, seed)
    predictions: list[ShipPrediction] = []
    fold0_replays: list[tuple[AsOfState, Replay]] = []

    for fold_index, test_ids in enumerate(folds):
        test_set = set(test_ids)
        train_states = [state for state in states if state.shipment_id not in test_set]
        test_states = [by_id[shipment_id] for shipment_id in test_ids]
        train_examples = [row for row in examples if row.shipment_id not in test_set]
        mean_delay = float(np.mean([state.label_final_delay_h for state in train_states]))

        duration_model = DurationModel()
        duration_model.fit(train_examples)
        flat_model = FlatDelayModel()
        flat_model.fit(train_states)
        flat_predictions = {
            state.shipment_id: float(value)
            for state, value in zip(test_states, flat_model.predict(test_states), strict=True)
        }

        for state in test_states:
            journey = predict_structured(duration_model, state)
            if fold_index == 0:
                fold0_replays.append((state, journey))
            predictions.append(
                ShipPrediction(
                    shipment_id=state.shipment_id,
                    route_id=state.route_id,
                    fold=fold_index,
                    actual_delay_h=state.label_final_delay_h,
                    schedule_h=0.0,
                    mean_h=mean_delay,
                    structured_h=journey.predicted_delay_h,
                    flat_h=flat_predictions[state.shipment_id],
                    customs_duration_error_h=_duration_error(
                        duration_model, state, "customs", truth_duration
                    ),
                    inland_duration_error_h=_duration_error(
                        duration_model, state, "inland_delivery", truth_duration
                    ),
                )
            )

    explained = _typical_explanation(fold0_replays)
    return Evaluation(predictions=tuple(predictions), explained=explained)


def _duration_error(
    model: DurationModel,
    state: AsOfState,
    stage: str,
    truth_duration: dict[tuple[str, str], float],
) -> float:
    view = next(item for item in state.remaining if item.stage == stage)
    predicted = model.predict_duration(view.stage, view.planned_duration_h, view.congestion_index)
    return predicted - truth_duration[(state.shipment_id, stage)]


def _typical_explanation(replays: list[tuple[AsOfState, Replay]]) -> ExplainedShipment:
    errors = [abs(journey.predicted_delay_h - state.label_final_delay_h) for state, journey in replays]
    median = float(np.median(errors))
    state, journey = min(
        replays,
        key=lambda pair: abs(abs(pair[1].predicted_delay_h - pair[0].label_final_delay_h) - median),
    )
    arrival = state.scheduled_arrival_utc + timedelta(hours=journey.predicted_delay_h)
    return ExplainedShipment(
        state=state,
        replay=journey,
        fold=0,
        predicted_arrival_utc=arrival,
    )


def _fold_maes(evaluation: Evaluation, field: str) -> list[float]:
    by_fold: dict[int, list[ShipPrediction]] = defaultdict(list)
    for row in evaluation.predictions:
        by_fold[row.fold].append(row)
    maes = []
    for fold in range(len(by_fold)):
        actual = np.array([row.actual_delay_h for row in by_fold[fold]])
        predicted = np.array([getattr(row, field) for row in by_fold[fold]])
        maes.append(_mae(actual, predicted))
    return maes


def _route_mae(evaluation: Evaluation, field: str) -> list[tuple[str, int, float]]:
    by_route: dict[str, list[ShipPrediction]] = defaultdict(list)
    for row in evaluation.predictions:
        by_route[row.route_id].append(row)
    rows = []
    for route in sorted(by_route):
        group = by_route[route]
        actual = np.array([row.actual_delay_h for row in group])
        predicted = np.array([getattr(row, field) for row in group])
        rows.append((route, len(group), _mae(actual, predicted)))
    return rows


def format_report(evaluation: Evaluation) -> str:
    actual = evaluation.column("actual_delay_h")
    methods = [
        ("Trust the schedule (predict 0 h)", "schedule_h"),
        ("Training-set mean delay", "mean_h"),
        ("Flat ridge on final delay", "flat_h"),
        ("Stage durations plus propagation", "structured_h"),
    ]
    header = [
        "# Model results",
        "",
        "Working note for the decision-time model. This is not the final 2–3 page submission.",
        "",
        "## Decision time",
        "",
        "The prediction is made when port of discharge has finished. Origin, loading, ocean, and discharge durations are observed. Customs and inland durations are predicted. Congestion indexes, including customs, are treated as known at this moment. The final delay is then the propagation rule from `eta_delay/propagate.py`, not a separate regression.",
        "",
        "Training uses five shipment-level folds and seed 20261111. A shipment's stages never sit on both sides of a fold. The training label for a stage duration is the true duration, including customs milestones that were not recorded. Those hidden timestamps are not inputs. At this decision time customs has not happened yet, so the missing milestone does not change the feature row.",
        "",
        "## Out-of-fold error",
        "",
        "Each shipment is scored once. MAE is mean absolute error of final delay in hours. Bias is the mean of prediction minus actual. Positive bias means the predicted arrival is too late.",
        "",
        "| Method | MAE (hours) | Bias (hours) | Fold MAE range |",
        "| --- | ---: | ---: | --- |",
    ]
    for label, field in methods:
        predicted = evaluation.column(field)
        fold_maes = _fold_maes(evaluation, field)
        header.append(
            f"| {label} | {_mae(actual, predicted):.2f} | {_bias(actual, predicted):.2f} | "
            f"{min(fold_maes):.2f} to {max(fold_maes):.2f} |"
        )

    customs_mae = float(np.mean(np.abs(evaluation.column("customs_duration_error_h"))))
    inland_mae = float(np.mean(np.abs(evaluation.column("inland_duration_error_h"))))
    header.extend(
        [
            "",
            f"On the two predicted stages, out-of-fold duration MAE is {customs_mae:.2f} hours for customs and {inland_mae:.2f} hours for inland delivery. That residual is mostly the noise term in the generator. The propagation step then turns those durations into a final delay.",
            "",
            "## MAE by route",
            "",
            "Out-of-fold mean absolute error in hours for the propagation model, next to the schedule baseline.",
            "",
            "| Route | Shipments | Schedule MAE | Propagation MAE |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    schedule_by_route = {route: mae for route, _, mae in _route_mae(evaluation, "schedule_h")}
    for route, count, mae in _route_mae(evaluation, "structured_h"):
        header.append(
            f"| {route} | {count} | {schedule_by_route[route]:.2f} | {mae:.2f} |"
        )

    header.extend(["", "## One shipment", ""])
    header.append(_explain_paragraph(evaluation.explained))
    header.extend(["", _explain_table(evaluation.explained), ""])
    header.extend(
        [
            "## What this does not show",
            "",
            "The duration model is a linear function of planned time and congestion, and the generator is the same kind of function plus noise. A low MAE on these 200 shipments means the propagation rule and the fit agree with the simulator. It is not evidence about real ports. Weather, revised schedules, and a port network are not in the data.",
            "",
            "The flat ridge is close on MAE, because lateness leaving discharge is already known and a linear model can use it. The gap is the explanation. The flat model has no slack rule, so it cannot say which stage's hours reached the consignee. The propagation model can: the contributions in the table add up to its predicted delay.",
            "",
        ]
    )
    return "\n".join(header)


def _label(stage: str) -> str:
    return stage.replace("_", " ")


def _fmt_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M UTC")


def _explain_paragraph(explained: ExplainedShipment) -> str:
    state = explained.state
    journey = explained.replay
    error = journey.predicted_delay_h - state.label_final_delay_h
    actual_arrival = state.scheduled_arrival_utc + timedelta(hours=state.label_final_delay_h)
    if abs(error) < 0.05:
        error_sentence = "The predicted delay matches the actual delay at this rounding."
    elif error < 0:
        error_sentence = (
            f"The predicted delay is {abs(error):.1f} hours shorter than the actual delay."
        )
    else:
        error_sentence = f"The predicted delay is {error:.1f} hours longer than the actual delay."
    reset = (
        f"The shipment was last back on schedule at {_label(journey.reset_stage)}, so earlier stages contribute 0 hours."
        if journey.reset_stage
        else "No earlier stage got the shipment back on schedule, so every stage from the start can contribute."
    )
    return "\n".join(
        [
            f"`{state.shipment_id}` on {state.route_id} ({state.origin} to {state.destination}) is a fold-{explained.fold} holdout. It was chosen because its absolute error is the closest in that fold to the median absolute error, so it is a typical miss rather than a best case.",
            "",
            f"At the decision time the carried delay leaving port of discharge is {state.carried_out_at_as_of_h:.1f} hours. {reset}",
            "",
            f"Predicted final delay is {journey.predicted_delay_h:.1f} hours. Actual final delay is {state.label_final_delay_h:.1f} hours. {error_sentence}",
            "",
            f"Scheduled arrival is {_fmt_time(state.scheduled_arrival_utc)}. Predicted arrival is {_fmt_time(explained.predicted_arrival_utc)}. Actual arrival is {_fmt_time(actual_arrival)}.",
            "",
            f"The contribution column sums to {journey.contribution_sum_h:.1f} hours, the predicted delay. A positive contribution is delay this stage still passes on. A negative contribution is allowance that absorbs earlier delay. A zero contribution is a stage whose effect was wiped out once the shipment returned to the schedule.",
        ]
    )


def _explain_table(explained: ExplainedShipment) -> str:
    lines = [
        "| Stage | Source | Duration (h) | Scheduled allowance (h) | Excess (h) | Carried out (h) | Contribution (h) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage in explained.replay.stages:
        lines.append(
            f"| {_label(stage.stage)} | {stage.source} | {stage.duration_h:.1f} | "
            f"{stage.scheduled_duration_h:.1f} | {stage.excess_h:.1f} | "
            f"{stage.carried_out_h:.1f} | {stage.contribution_h:.1f} |"
        )
    return "\n".join(lines)


def write_outputs(evaluation: Evaluation) -> None:
    REPORT_PATH.write_text(format_report(evaluation), encoding="utf-8")
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "shipment_id",
        "route_id",
        "fold",
        "actual_delay_h",
        "schedule_h",
        "mean_h",
        "structured_h",
        "flat_h",
    ]
    with OOF_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in evaluation.predictions:
            writer.writerow({field: getattr(row, field) for field in fields})


def main() -> None:
    journeys, shipments = generate_dataset()
    evaluation = evaluate(journeys, shipments)
    write_outputs(evaluation)
    actual = evaluation.column("actual_delay_h")
    structured = evaluation.column("structured_h")
    schedule = evaluation.column("schedule_h")
    print(
        f"out-of-fold MAE hours: schedule {_mae(actual, schedule):.2f}, "
        f"propagation {_mae(actual, structured):.2f}"
    )
    print(f"explained shipment: {evaluation.explained.state.shipment_id}")
    print(f"wrote {REPORT_PATH.relative_to(REPORT_PATH.parents[1])} and data/oof_predictions.csv")


if __name__ == "__main__":
    main()
