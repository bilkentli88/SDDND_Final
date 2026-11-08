# Experiment 2: model and observation-design check

Prepared 19 October 2026, before the new noisy learning experiment. This note checks the fixed matched-state algebra used by the manuscript. It does not assert identifiability for an unrestricted complete-trajectory observation operator.

## Selected mechanism

Use the synthetic state-dependent extension of delayed logistic growth

\[
\dot x(t)=\gamma x(t)\left[1-\frac{x(t-\tau^\star(x(t)))}{K}\right],
\qquad \gamma=1.1,\quad K=1.5,
\]

\[
\tau^\star(s)=0.28+0.07\tanh(2(s-0.8)),\qquad I_\tau=[0.15,0.50].
\]

The delayed logistic equation supplies the population-growth interpretation. The particular state-dependent delay and numerical parameters above are synthetic design choices, not a biologically calibrated model. For background, see Rebenda and Smarda, *An efficient numerical approach for delayed logistic models*, [arXiv:1610.01528](https://arxiv.org/abs/1610.01528), especially their constant-delay logistic equation (2). That source motivates the base equation; it does not supply our state-dependent law or the theorem below.

The manuscript's affine delayed-state form holds with

\[
a^\star(s)=1.1s,\qquad b^\star(s)=-\frac{1.1}{1.5}s.
\]

On the target state interval \(S_0=[0.45,1.20]\), \(|b^\star(s)|\ge0.33\). The gain is negative, so the old positive-softplus gain parameterization would misspecify the learner. Both P3 and P4 therefore use \(-[0.05+\operatorname{softplus}(g_\theta(s))]\). Their networks remain otherwise matched. The known negative sign is an additional modeling prior shared by both neural methods; the projective uniqueness calculation itself allows either nonzero sign.

## Four informative histories

For each matched initial state \(s\), prescribe

\[
\phi_{s,j}(\theta)=s+\epsilon\{c_j\sin(\omega\theta)+d_j[1-\cos(\omega\theta)]\},
\quad -0.5\le\theta\le0,
\]

with \(\epsilon=0.18\), \(\omega=\pi\),

\[
c=(-1,0,1,0)^\top,\qquad d=(0,1,0,-1)^\top.
\]

All histories end at \(s\). Each history has only one nonzero coefficient, so it lies in \([s-0.18,s+0.18]\) on this interval. Across training and test states, all histories are strictly positive, at least 0.27. The vectors \(c,d\) have zero mean, are orthogonal, and have squared norm two.

At the matched initial events, the centered delayed-history signature is

\[
w_s(r)=-\epsilon\sin(\omega r)c+\epsilon[1-\cos(\omega r)]d.
\]

It is independent of \(s\), and

\[
\|w_s(r)\|=2\sqrt2\epsilon\sin(\omega r/2)>0
\quad(r\in[0.15,0.50]).
\]

The ratio of its two coefficients, with sign removed, is

\[
\frac{1-\cos(\omega r)}{\sin(\omega r)}=\tan(\omega r/2).
\]

This function is strictly increasing because \(0<\omega r/2\le\pi/4\). Orthogonality of \(c,d\) therefore proves that \(r\mapsto\operatorname{span}\{w_s(r)\}\) is injective. This is an analytic projective-diversity proof, not a sampled numerical check.

If \(y_s=b(s)w_s(\tau(s))\), define \(C=c^\top y_s/2\) and \(D=d^\top y_s/2\). Then \(C\ne0\) and

\[
\tau(s)=\frac{2}{\omega}\arctan(-D/C).
\]

The experiment uses grid-based profiles estimated from observations, not this formula applied to oracle derivatives. The formula is an algebraic validation check only.

## Ideal stability calculation

For the Euclidean-distance profile \(P_s(r)=\operatorname{dist}(y_s,\operatorname{span}\{w_s(r)\})\), the angle between the two history lines gives

\[
P_s(r)=|b(s)|\|w_s(\tau(s))\|
\left|\sin\left(\frac{\omega(r-\tau(s))}{2}\right)\right|.
\]

Writing \(\Delta=0.50-0.15\), an ideal uniform separation bound over \(S_0\) is

\[
P_s(r)\ge\underline\kappa|r-\tau(s)|,
\quad
\underline\kappa=0.33\,[2\sqrt2\epsilon\sin(0.15\omega/2)]
\frac{\sin(\omega\Delta/2)}{\Delta}>0.
\]

This is a bound for the **unsquared Euclidean-distance profile**. The implemented anchor objective is its normalized square, \(P_s(r)^2/4\); a linear separation bound cannot be transferred unchanged to that squared quantity. Gaussian observation noise also supplies no deterministic uniform error envelope merely from knowing its standard deviation. The noisy experiment consequently makes no deterministic-certification or calibrated-confidence claim.

## Positive solutions, regularity, and the initial endpoint

For a positive history, the logistic equation has the multiplicative form \(\dot x=xg\) with continuous bounded coefficient on each finite method-of-steps interval. Since \(\tau\ge0.15\), the delayed values in a sufficiently short next interval are already defined. The known past is Lipschitz; composing it with the smooth state-dependent lag yields a locally Lipschitz right-hand side in the current state. Iteration gives a unique forward solution. Positivity follows from the exponential representation on each interval.

On \([0,1]\), \(x(t)\le x(0)e^{1.1t}\), so a common upper bound including the prescribed past is \(M=\max(1.38,1.20e^{1.1})\). A positive lower bound is \(m=0.45\exp\{1.1(1-M/1.5)\}\). Thus the solution stays in a positive compact state domain and the gain remains nonzero there. These are analytic bounds for the true equation, not guaranteed bounds for an arbitrary learned model.

The histories need not satisfy the derivative compatibility condition at time zero. We explicitly use **right-hand initial responses** \(v_{s,j}=D^+x_{s,j}(0)\). The forward DDE gives \(v_{s,j}=a(s)+b(s)\phi_{s,j}(-\tau(s))\), so the fixed matched-slope algebra and projective proof apply. We do not assert a globally C1 concatenation across the initial-history junction or invoke the separate C1,1 monotone-trajectory assumptions of the unrestricted compensation theorem. This endpoint convention must be explicit when the experiment is added to the article.

## Observation coverage and negative control

The experiment observes four matched training initial states and three held-out initial states. The analytic family is informative for any \(s\in S_0\), but observations at four training states do not by themselves certify the learned delay at all 201 evaluation states. The dense-grid metric measures interpolation against synthetic truth. No dense-grid certification is claimed.

The degenerate control replaces all four histories at a state by \(\phi_{s,j}(\theta)=s\). Then \(q_s(r)=s\mathbf1\), \(w_s(r)=0\), and the matched-initial-state profile is flat. Such anchors are rejected by a history-only geometry gate. Direct methods and anchor-only networks abstain. P3 consequently reduces to P4, and their equality under paired initialization is checked. This shows failure of the specified matched-state mechanism, not universal non-identifiability from every possible full-trajectory experiment.
