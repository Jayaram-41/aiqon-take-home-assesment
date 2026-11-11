"""The generator matches the hand-worked journey and the dataset rules."""

from datetime import UTC, datetime

from eta_delay.checks import check_dataset, propagation_errors
from eta_delay.generate import generate_dataset, hand_example


def test_hand_example_final_delay_is_26_hours():
    stages, shipment = hand_example()
    # Discharge congestion of 1 adds 40h against a 10h buffer, so 30h leaves
    # the port. Customs congestion of 0.5 adds 8h against a 6h buffer and
    # starts 30h late, so 32h leaves customs. Inland buffer is 6h.
    # 32 - 6 = 26 hours late at the consignee.
    assert shipment.final_delay_h == 26.0
    assert shipment.scheduled_arrival_utc == datetime(2026, 1, 26, 16, tzinfo=UTC)
    assert shipment.actual_arrival_utc == datetime(2026, 1, 27, 18, tzinfo=UTC)
    assert [stage.carried_out_h for stage in stages] == [-2.0, -8.0, -12.0, 30.0, 32.0, 26.0]
    assert propagation_errors(stages) == []


def test_dataset_has_200_shipments_and_a_real_delay_mix():
    journeys, shipments = generate_dataset()
    report = check_dataset(journeys, shipments)
    assert report.ok, report.failures
    assert report.n_shipments == 200
    assert report.n_stage_rows == 1200


def test_same_seed_reproduces_the_file():
    first, _ = generate_dataset()
    second, _ = generate_dataset()
    assert [stage.truth_end_utc for stage in first] == [stage.truth_end_utc for stage in second]
    assert [stage.carried_out_h for stage in first] == [stage.carried_out_h for stage in second]


def test_hidden_customs_keeps_the_truth_and_blanks_the_observation():
    journeys, shipments = generate_dataset()
    hidden = next(shipment for shipment in shipments if not shipment.customs_observed)
    customs = next(
        stage
        for stage in journeys
        if stage.shipment_id == hidden.shipment_id and stage.stage == "customs"
    )
    assert customs.truth_end_utc is not None
    assert customs.observed_end_utc is None
    assert customs.is_observed is False
