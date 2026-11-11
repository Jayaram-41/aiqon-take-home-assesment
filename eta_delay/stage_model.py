"""Predict remaining stage durations, then hand them to the propagation rule.

Each stage has its own ridge regression. The features are the planned
duration and the congestion index. The buffer is not a regression feature.
Propagation applies the buffer later, when it decides whether delay carries
forward.

Planned duration is constant for some stages. Those fits use only the
congestion index, plus an intercept. Ridge here is a small linear model:
with about 160 training rows the penalty barely moves an ordinary
least-squares fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from eta_delay.definitions import MIN_DURATION_H, ROUTES, STAGES
from eta_delay.features import AsOfState, DurationExample
from eta_delay.propagate import StageInput, replay, Replay

RIDGE_ALPHA = 1.0


@dataclass
class _StageFit:
    names: tuple[str, ...]
    scaler: StandardScaler | None
    model: Ridge | None
    mean_h: float


class DurationModel:
    def __init__(self, alpha: float = RIDGE_ALPHA) -> None:
        self.alpha = alpha
        self._fits: dict[str, _StageFit] = {}

    def fit(self, examples: list[DurationExample]) -> None:
        self._fits = {}
        for stage in STAGES:
            rows = [example for example in examples if example.stage == stage]
            if not rows:
                raise ValueError(f"no training rows for {stage}")
            labels = np.array([row.label_duration_h for row in rows], dtype=float)
            names, matrix = _design(rows)
            if matrix is None:
                self._fits[stage] = _StageFit(names=(), scaler=None, model=None, mean_h=float(np.mean(labels)))
                continue
            scaler = StandardScaler()
            model = Ridge(alpha=self.alpha)
            model.fit(scaler.fit_transform(matrix), labels)
            self._fits[stage] = _StageFit(
                names=names,
                scaler=scaler,
                model=model,
                mean_h=float(np.mean(labels)),
            )

    def predict_duration(self, stage: str, planned_duration_h: float, congestion_index: float) -> float:
        fit = self._fits[stage]
        if fit.model is None or fit.scaler is None:
            return max(MIN_DURATION_H, fit.mean_h)
        values = {
            "planned_duration_h": planned_duration_h,
            "congestion_index": congestion_index,
        }
        matrix = np.array([[values[name] for name in fit.names]], dtype=float)
        predicted = float(fit.model.predict(fit.scaler.transform(matrix))[0])
        return max(MIN_DURATION_H, predicted)


class FlatDelayModel:
    """Predict final delay directly from the decision-time summary.

    This is the comparison model. It does not apply slack stage by stage.
    """

    def __init__(self, alpha: float = RIDGE_ALPHA) -> None:
        self.alpha = alpha
        self._scaler = StandardScaler()
        self._model = Ridge(alpha=alpha)
        self._reference_route = ROUTES[0].route_id

    def fit(self, states: list[AsOfState]) -> None:
        labels = np.array([state.label_final_delay_h for state in states], dtype=float)
        self._model.fit(self._scaler.fit_transform(self._matrix(states)), labels)

    def predict(self, states: list[AsOfState]) -> np.ndarray:
        return self._model.predict(self._scaler.transform(self._matrix(states)))

    def _matrix(self, states: list[AsOfState]) -> np.ndarray:
        rows = []
        for state in states:
            numeric = [
                state.carried_out_at_as_of_h,
                state.congestion("customs"),
            ]
            dummies = [
                1.0 if state.route_id == route.route_id else 0.0 for route in ROUTES[1:]
            ]
            rows.append(numeric + dummies)
        return np.array(rows, dtype=float)


def _design(rows: list[DurationExample]) -> tuple[tuple[str, ...], np.ndarray | None]:
    planned = np.array([row.planned_duration_h for row in rows], dtype=float)
    congestion = np.array([row.congestion_index for row in rows], dtype=float)
    columns: list[np.ndarray] = []
    names: list[str] = []
    if float(np.std(planned)) > 1e-8:
        columns.append(planned)
        names.append("planned_duration_h")
    if float(np.std(congestion)) > 1e-8:
        columns.append(congestion)
        names.append("congestion_index")
    if not columns:
        return (), None
    return tuple(names), np.column_stack(columns)


def predict_structured(model: DurationModel, state: AsOfState) -> Replay:
    """Observed durations through discharge, predicted durations after that."""
    inputs: list[StageInput] = [
        StageInput(
            stage=view.stage,
            scheduled_duration_h=view.scheduled_duration_h,
            duration_h=float(view.observed_duration_h),
            source="observed",
        )
        for view in state.completed
    ]
    for view in state.remaining:
        inputs.append(
            StageInput(
                stage=view.stage,
                scheduled_duration_h=view.scheduled_duration_h,
                duration_h=model.predict_duration(
                    view.stage,
                    view.planned_duration_h,
                    view.congestion_index,
                ),
                source="predicted",
            )
        )
    return replay(inputs)
