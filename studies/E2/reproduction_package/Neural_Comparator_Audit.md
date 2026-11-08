# Neural-comparator compatibility check

This is a focused compatibility audit, not a completed external-method benchmark. It is separate from the fixed Experiment 2 protocol. No extra method was added to that experiment after its outcomes began to appear.

## Monsel et al. (2024)

The [PMLR publication](https://proceedings.mlr.press/v255/monsel24a.html) introduces a neural framework that handles time- and state-dependent delayed dynamics and links its implementation. The released state-dependent training script was inspected at repository commit `046379c60c40c8bebe707f5d4c11f471092f9782`.

In [`state_dependent.py`](https://github.com/thibmonsel/Time-and-State-Dependent-Neural-Delay-Differential-Equations/blob/046379c60c40c8bebe707f5d4c11f471092f9782/state_dependent.py), the DDE branch supplies the analytic delay `0.5*cos(y[0])` to the solver. [`models.py`](https://github.com/thibmonsel/Time-and-State-Dependent-Neural-Delay-Differential-Equations/blob/046379c60c40c8bebe707f5d4c11f471092f9782/models.py) stores that delay object and learns the vector field. This published configuration therefore does not automatically provide a comparator for discovering an unknown state-dependent delay. That observation concerns the inspected configuration, not a claim that every possible extension of the framework fixes delays.

Its appropriate role is a neural delayed-dynamics/prediction reference. An unknown-delay extension would need explicit implementation and disclosure, rather than being presented as the unchanged published method.

## Zhang, Zhu and Lin (2026)

The publisher's [GNDDE article](https://doi.org/10.1063/5.0325998), *A general framework for neural delay differential equations with various delay types*, explicitly describes state-dependent delay functions, adaptive delay identification, and simulation-free regression using preprocessed targets. It is a stronger candidate for the unknown-delay comparison than a method learning a single constant lag.

The available publisher abstract supports that candidacy; it does not settle all implementation details. Before running a comparison, inspect the full method's target preprocessing, delay parameterization, history access, model-based versus model-free assumptions, and code availability. Then fix a common observation budget and tuning protocol. Access to exact synthetic derivatives must not be substituted for the noisy observations supplied to P3/P4. No claim of a reproduced GNDDE result or demonstrated superiority over GNDDE is made here.

## Consequence for the revision

P4 remains an architecture-matched rollout control, not a faithful reimplementation of either external publication. Experiment 2 tests generality within the specified scalar setting and cannot replace a compatible external-method comparison. Complete the GNDDE compatibility/reproduction assessment alongside the remaining resolution study before final comparative claims are written.
