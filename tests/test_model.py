"""Decision-time features, the slack attribution, and the held-out score."""

from dataclasses import replace
from datetime import timedelta

from eta_delay.evaluate import evaluate
from eta_delay.features import DURATION_FEATURE_FIELDS, build_as_of_states, duration_examples
from eta_delay.generate import generate_dataset, hand_example
from eta_delay.propagate import StageInput, replay
from eta_delay.stage_model import DurationModel, predict_structured


def test_hand_example_contributions_sum_to_26_hours():
    stages, shipment = hand_example()
    journey = replay(
        [
            StageInput(
                stage=stage.stage,
                scheduled_duration_h=stage.scheduled_duration_h,
                duration_h=stage.actual_duration_h,
                source="observed",
            )
            for stage in stages
        ]
    )
    assert [round(stage.contribution_h, 6) for stage in journey.stages] == [0.0, 0.0, 0.0, 30.0, 2.0, -6.0]
    assert journey.predicted_delay_h == 26.0
    assert journey.contribution_sum_h == shipment.final_delay_h
    assert journey.reset_stage == "ocean_transit"


def test_true_durations_reconstruct_every_shipment():
    journeys, shipments = generate_dataset()
    by_id: dict[str, list] = {}
    for stage in journeys:
        by_id.setdefault(stage.shipment_id, []).append(stage)
    for shipment in shipments:
        stages = sorted(by_id[shipment.shipment_id], key=lambda stage: stage.stage_order)
        journey = replay(
            [
                StageInput(
                    stage=stage.stage,
                    scheduled_duration_h=stage.scheduled_duration_h,
                    duration_h=stage.actual_duration_h,
                    source="observed",
                )
                for stage in stages
            ]
        )
        assert abs(journey.predicted_delay_h - shipment.final_delay_h) < 1e-6
        assert abs(journey.contribution_sum_h - journey.predicted_delay_h) < 1e-6


def test_future_truth_is_not_a_decision_time_feature():
    journeys, shipments = generate_dataset()
    states = {state.shipment_id: state for state in build_as_of_states(journeys, shipments)}
    examples = duration_examples(journeys)
    assert set(examples[0].features()) == set(DURATION_FEATURE_FIELDS)

    mutated = []
    for stage in journeys:
        if stage.stage == "port_of_discharge":
            mutated.append(
                replace(
                    stage,
                    actual_duration_h=1.0,
                    noise_h=99.0,
                    carried_out_h=99.0,
                    truth_end_utc=stage.truth_start_utc + timedelta(hours=1),
                )
            )
        elif stage.stage in ("customs", "inland_delivery"):
            mutated.append(replace(stage, actual_duration_h=1.0, noise_h=99.0, carried_out_h=99.0))
        else:
            mutated.append(stage)

    mutated_states = {state.shipment_id: state for state in build_as_of_states(mutated, shipments)}
    for shipment_id, state in states.items():
        other = mutated_states[shipment_id]
        assert state.carried_out_at_as_of_h == other.carried_out_at_as_of_h
        assert state.completed == other.completed
        assert [view.congestion_index for view in state.remaining] == [
            view.congestion_index for view in other.remaining
        ]
        assert all(view.observed_duration_h is None for view in state.remaining)


def test_changing_the_label_does_not_change_the_prediction():
    journeys, shipments = generate_dataset()
    train = duration_examples(journeys[: 6 * 160])
    model = DurationModel()
    model.fit(train)
    state = build_as_of_states(journeys, shipments)[-1]
    flipped = replace(state, label_final_delay_h=state.label_final_delay_h + 100.0)
    original = predict_structured(model, state)
    other = predict_structured(model, flipped)
    assert original.predicted_delay_h == other.predicted_delay_h


def test_propagation_model_beats_the_schedule_baseline():
    journeys, shipments = generate_dataset()
    evaluation = evaluate(journeys, shipments)
    actual = evaluation.column("actual_delay_h")
    schedule_mae = float(abs(actual - evaluation.column("schedule_h")).mean())
    mean_mae = float(abs(actual - evaluation.column("mean_h")).mean())
    structured_mae = float(abs(actual - evaluation.column("structured_h")).mean())
    assert len(evaluation.predictions) == 200
    assert len({row.shipment_id for row in evaluation.predictions}) == 200
    assert structured_mae < schedule_mae
    assert structured_mae < mean_mae
    explained = evaluation.explained
    assert abs(explained.replay.contribution_sum_h - explained.replay.predicted_delay_h) < 1e-6
    assert explained.predicted_arrival_utc == explained.state.scheduled_arrival_utc + timedelta(
        hours=explained.replay.predicted_delay_h
    )
