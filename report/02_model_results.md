# Model results

Working note for the decision-time model. This is not the final 2–3 page submission.

## Decision time

The prediction is made when port of discharge has finished. Origin, loading, ocean, and discharge durations are observed. Customs and inland durations are predicted. Congestion indexes, including customs, are treated as known at this moment. The final delay is then the propagation rule from `eta_delay/propagate.py`, not a separate regression.

Training uses five shipment-level folds and seed 20261111. A shipment's stages never sit on both sides of a fold. The training label for a stage duration is the true duration, including customs milestones that were not recorded. Those hidden timestamps are not inputs. At this decision time customs has not happened yet, so the missing milestone does not change the feature row.

## Out-of-fold error

Each shipment is scored once. MAE is mean absolute error of final delay in hours. Bias is the mean of prediction minus actual. Positive bias means the predicted arrival is too late.

| Method | MAE (hours) | Bias (hours) | Fold MAE range |
| --- | ---: | ---: | --- |
| Trust the schedule (predict 0 h) | 12.00 | -9.97 | 9.85 to 13.61 |
| Training-set mean delay | 10.45 | -0.00 | 10.28 to 10.71 |
| Flat ridge on final delay | 4.76 | -0.05 | 4.30 to 5.24 |
| Stage durations plus propagation | 4.04 | -0.07 | 3.30 to 4.62 |

On the two predicted stages, out-of-fold duration MAE is 2.94 hours for customs and 2.97 hours for inland delivery. That residual is mostly the noise term in the generator. The propagation step then turns those durations into a final delay.

## MAE by route

Out-of-fold mean absolute error in hours for the propagation model, next to the schedule baseline.

| Route | Shipments | Schedule MAE | Propagation MAE |
| --- | ---: | ---: | ---: |
| CNSHA-NLRTM | 56 | 10.26 | 4.54 |
| INNSA-AEJEA | 49 | 10.91 | 3.41 |
| KRPUS-DEHAM | 52 | 13.16 | 3.56 |
| SGSIN-USLAX | 43 | 14.12 | 4.68 |

## One shipment

`SHP-004` on INNSA-AEJEA (INNSA to AEJEA) is a fold-0 holdout. It was chosen because its absolute error is the closest in that fold to the median absolute error, so it is a typical miss rather than a best case.

At the decision time the carried delay leaving port of discharge is 10.9 hours. The shipment was last back on schedule at ocean transit, so earlier stages contribute 0 hours.

Predicted final delay is 6.9 hours. Actual final delay is 10.4 hours. The predicted delay is 3.4 hours shorter than the actual delay.

Scheduled arrival is 2026-02-26 19:00 UTC. Predicted arrival is 2026-02-27 01:56 UTC. Actual arrival is 2026-02-27 05:23 UTC.

The contribution column sums to 6.9 hours, the predicted delay. A positive contribution is delay this stage still passes on. A negative contribution is allowance that absorbs earlier delay. A zero contribution is a stage whose effect was wiped out once the shipment returned to the schedule.

| Stage | Source | Duration (h) | Scheduled allowance (h) | Excess (h) | Carried out (h) | Contribution (h) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| origin departure | observed | 9.0 | 10.0 | -1.0 | -1.0 | 0.0 |
| port of loading | observed | 25.9 | 36.0 | -10.1 | -10.1 | 0.0 |
| ocean transit | observed | 119.1 | 132.0 | -12.9 | -12.9 | 0.0 |
| port of discharge | observed | 50.9 | 40.0 | 10.9 | 10.9 | 10.9 |
| customs | predicted | 28.7 | 26.0 | 2.7 | 13.7 | 2.7 |
| inland delivery | predicted | 29.3 | 36.0 | -6.7 | 6.9 | -6.7 |

## What this does not show

The duration model is a linear function of planned time and congestion, and the generator is the same kind of function plus noise. A low MAE on these 200 shipments means the propagation rule and the fit agree with the simulator. It is not evidence about real ports. Weather, revised schedules, and a port network are not in the data.

The flat ridge is close on MAE, because lateness leaving discharge is already known and a linear model can use it. The gap is the explanation. The flat model has no slack rule, so it cannot say which stage's hours reached the consignee. The propagation model can: the contributions in the table add up to its predicted delay.
