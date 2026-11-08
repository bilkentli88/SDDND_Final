# Complete two-stage control for the archived E1--E3 studies

The aim is to complete the previously missing forecasting comparison. Estimate
the delay from the observed profile anchors, freeze the estimated delay
function, then learn the instantaneous and gain fields from the same noisy
trajectories. This tests the incremental value of joint fitting for a complete
predictive model. It does not aim to make the joint method win.

This is an extension after inspection of E1--E3 outcomes. The existing datasets,
profiles and anchor-only checkpoints are reused. It is not an independent
replication, blind confirmation or public preregistration. No test metric or
generating delay is used to tune the new control. The implementation and this
protocol are frozen before scientific stage-two fits; archive checkpoint
replay and four-update implementation fixtures are checked first.

## Conditions and methods

- E1: the original scalar mechanism, noise SD 0.0005.
- E2: informative logistic histories at SDs 0.0005 and 0.0015.
- E3: all four informative logistic observation/training-step conditions,
  observation spacings 0.05/0.025 crossed with Euler steps 0.025/0.0125.
- Every condition reuses all five base-noise datasets and all three nested
  paired optimizer seeds. The E2 noise conditions and E3 resolution conditions
  retain their original pairing. These are not seven independent sets of five.
- T1 uses the exact saved A1 delay branch (280 anchor-only updates); T2 uses
  the saved A2 branch (1,200 anchor-only updates). Both receive 280 additional
  field-training updates, 90 through time 0.5 and 190 through time one.
- The a/b networks start from the original paired random initialization,
  before any trajectory fitting. Neither is initialized from P3/P4 fitted
  fields. Architectures, sign priors, optimizer rate, weight decay, gradient
  clipping, observations and rollout scheme follow the corresponding study.
- Stage two minimizes the same weighted trajectory loss. Delay anchor and
  delay-smoothness penalties are constant when this function is frozen, so
  are omitted from the field objective. Delay parameters have no gradients
  or optimizer updates; differentiation of tau(x) with respect to the current
  state is retained for backpropagation through the rollout.
- P3 and P4 are replayed from the exact archived checkpoints in a common
  evaluation harness. E2/E3 rollout source files are copied unchanged. The E1
  discrete-history Euler adapter is checked against all archived model scores;
  its adaptation and measured replay differences are reported explicitly.
- Degenerate-history cases have no admissible observed-anchor estimate for
  this pipeline and are not assigned a fabricated two-stage fit. Their prior
  structural abstention results remain unchanged. E4 is not extended here.

There are 105 paired study/condition/dataset/optimizer combinations and two
new field fits each: 210 new fits. Stage-one fits are reused, not counted as
new training. Total parameter-update stages are 560 for T1 and 1,480 for T2,
versus 280 for P3/P4. This grants the two-stage estimators additional work;
it is not an equal-total-compute benchmark. Record field-training wall time
and the archived stage-one time separately; do not infer speed ratios across
the old and current execution environments.

## Outcomes and reporting

The primary new outcome is clean held-out trajectory MSE, using the archived
study's test states, observation times and scoring convention. E3's primary
score uses Euler step 0.00625 on the 0.025 evaluation grid; native-step scores
are secondary. E1 retains its Euler-generated reference and E2 its RK4-based
reference. No cross-protocol pool is used as a single effect estimate.

Within each dataset, first take the median over the three optimizer repeats.
For each condition, report the median paired P3/T1 and P3/T2 forecast-MSE
ratios and wins among all five datasets; ratios below one favor joint fitting.
Also report absolute dataset-level scores, P4 comparisons and delay RMSE.
T1/T2 delay curves must be identical to their A1/A2 inputs, so their delay
scores do not supply new independent delay-recovery evidence. Do not choose
a stage-one method using test performance or report only its favorable cells.

This extension introduces no binary success threshold or calibrated confidence
claim. Its purpose is a complete comparison, including unfavorable findings.
Report all failures without automatic scientific retries; incomplete cells
receive no complete-case superiority verdict. A numerical implementation bug
must be diagnosed, fixed and disclosed before continuing, with any changed
freeze recorded.

## Validation and archival

Check archived delay curves and forecast metrics before starting the new
fits; record any E1 adapter discrepancy. Verify paired initial delay hashes,
exact frozen-branch hashes, unchanged grid predictions/delay RMSE, updated
fields and retained state derivatives. Save every checkpoint and record as
it completes. Recompute all final summaries and replay forecast scores from
the saved checkpoints. These checks establish consistency, not independent
scientific replication. Retain original archive hashes, software versions,
source hashes, timings, raw inputs and complete machine-readable records.
