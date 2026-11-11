# Definitions and synthetic journeys

Working note for the dataset. This is not the final 2–3 page submission. The decision-time model and its error are in [02_model_results.md](02_model_results.md).

## How delay is defined

All clocks are UTC.

The scheduled final arrival is the scheduled end of inland delivery. The actual final arrival is the true end of inland delivery. Ground truth does not depend on whether an intermediate milestone was recorded.

```text
final_delay_hours = actual_final_arrival - scheduled_final_arrival
```

The unit is hours. Positive means late. Zero or negative means the shipment is on time, including early. There is no separate "delayed" yes/no label.

The later model is allowed to see actuals only through port of discharge. Customs and inland delivery are still in the future at that moment. That decision time is stored on every shipment as `as_of_stage`.

## Stages

Every shipment has the same six stages, in this order:

| Order | Stage | What the hours represent |
| --- | --- | --- |
| 1 | Origin departure | Gate to vessel cutoff |
| 2 | Port of loading | Dwell before the vessel sails |
| 3 | Ocean transit | Sea leg |
| 4 | Port of discharge | Dwell after the vessel arrives |
| 5 | Customs | Clearance after discharge |
| 6 | Inland delivery | Final move to the consignee |

The schedule for a stage is planned work time plus a buffer. The buffer is slack. Planned work depends on the route for loading, ocean, discharge, and inland. Origin is always 8 hours. Customs is always 20 hours.

| Stage | Buffer (hours) | Extra hours if congestion is 1 |
| --- | --- | --- |
| Origin departure | 2 | 0 |
| Port of loading | 8 | 20 |
| Ocean transit | 12 | 0 |
| Port of discharge | 10 | 40 |
| Customs | 6 | 16 |
| Inland delivery | 6 | 0 |

Congestion on loading, discharge, and customs is drawn uniformly from 0 to 1. The other stages have congestion 0. Weather is not in this dataset.

| Route | Loading | Ocean | Discharge | Inland |
| --- | ---: | ---: | ---: | ---: |
| CNSHA–NLRTM | 30 | 720 | 40 | 36 |
| SGSIN–USLAX | 22 | 504 | 48 | 28 |
| INNSA–AEJEA | 28 | 120 | 30 | 30 |
| KRPUS–DEHAM | 26 | 768 | 42 | 40 |

Those route hours are planned work, before the buffer.

## How a delay is introduced

The schedule is built first and does not move. Realized duration is applied after that.

```text
scheduled_duration = planned_duration + buffer
congestion_effect = congestion_index * sensitivity
raw_duration = planned_duration + congestion_effect + noise
actual_duration = max(1 hour, raw_duration)
```

Noise is a normal draw with mean 0. The standard deviations are 2, 5, 6, 6, 4, and 4 hours for the six stages in order.

Propagation:

```text
carried_in = 0 for the first stage
truth_start = scheduled_start + carried_in
truth_end = truth_start + actual_duration
carried_out = truth_end - scheduled_end
next carried_in = max(0, carried_out)
```

A stage never starts before its scheduled start. Finishing early does not pull the next milestone forward. That early time is lost. Delay carries forward only when `carried_out` is positive, and the next buffer can still absorb it.

Final delay is `carried_out` of inland delivery. That is the ground truth.

About 18 percent of shipments leave the customs observation blank. The true customs timestamps stay in `truth_start_utc` and `truth_end_utc`, because the clearance still happened. `observed_start_utc` and `observed_end_utc` are empty, and `is_observed` is false. Downstream stages keep their true times. A later feature builder must use the observed columns, not the truth columns, for any stage it treats as unseen.

Seed `20261111` draws 200 shipments. Booked departures fall uniformly in the 90 days from 2026-01-01, on a random hour, in UTC. The same seed rewrites the same file.

## Worked example

`EX-001` is not one of the 200 rows. It uses the same function with every noise term set to 0, discharge congestion 1, customs congestion 0.5, and route INNSA–AEJEA, departing 2026-01-15 00:00 UTC.

| Stage | Carried in | Actual duration | Scheduled duration | Carried out |
| --- | ---: | ---: | ---: | ---: |
| Origin departure | 0 | 8 | 10 | −2 |
| Port of loading | 0 | 28 | 36 | −8 |
| Ocean transit | 0 | 120 | 132 | −12 |
| Port of discharge | 0 | 70 | 40 | +30 |
| Customs | 30 | 28 | 26 | +32 |
| Inland delivery | 32 | 30 | 36 | +26 |

Discharge congestion adds 40 hours against a 10-hour buffer, so 30 hours leave the port. Customs adds 8 hours (`0.5 × 16`) against a 6-hour buffer and starts 30 hours late, so 32 hours leave customs. The inland buffer is 6 hours. Final delay is 26 hours.

Scheduled arrival is 2026-01-26 16:00 UTC. Actual arrival is 2026-01-27 18:00 UTC. The difference is 26 hours. The test `test_hand_example_final_delay_is_26_hours` locks this arithmetic.

## What the 200 shipments contain

Regenerated with seed `20261111`:

- 200 shipments, 1,200 stage rows, all four routes (56, 49, 52, and 43).
- 48 shipments on time (`final_delay_h <= 0`) and 152 late.
- Final delay from −11.9 hours to 49.1 hours. Median 7.2 hours, mean 10.0 hours.
- Customs observation missing on 27 shipments.
- 42 of the on-time shipments were late at an earlier stage. Slack absorbed that lateness before the consignee.

`SHP-008` (KRPUS–DEHAM) is one of those. Loading finishes about 8.4 hours behind its scheduled end. The ocean buffer absorbs that, and the vessel is back ahead of the ocean schedule. Customs then finishes about 3.7 hours late. Inland slack absorbs it, and the shipment arrives about 2.5 hours early.

These counts describe this generator and this seed. They are not a claim about real ports. The buffers are tight relative to discharge congestion, which is why most shipments are late.

## Files

`data/shipments.csv` has one row per shipment: route, booked departure, both arrivals, final delay, the three congestion indexes, and whether customs was observed.

`data/journeys.csv` has one row per stage. `truth_*` timestamps are the ground truth, including hidden customs. `observed_*` timestamps are what a later model may treat as recorded. `carried_in_h` and `carried_out_h` are the propagation state in hours.

Using `truth_*` for a stage after port of discharge, or for an unobserved customs milestone, would leak the answer into tomorrow's model.

## Assumptions stated for the report

- Positive delay means late. The unit is hours. Clocks are UTC.
- The schedule is not revised after booking.
- A milestone cannot occur before its scheduled time.
- Slack is a fixed buffer per stage, shared by every route.
- Congestion lengthens loading, discharge, and customs only.
- Weather, strikes, rollovers, and blank sailings are not simulated.
- Missing data is limited to customs timestamps. The physical event still occurs.
- The 200 journeys are synthetic. The seed and the rules above are the full description of how delay was introduced and how ground truth was defined.
