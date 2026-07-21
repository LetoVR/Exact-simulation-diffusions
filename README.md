# Exact Simulation of Diffusion Processes

Unbiased Monte-Carlo estimation of path functionals `E[f(X)]` for
diffusions solving `dX_t = b(X_t)dt + σ(X_t)dW_t`, with application to
option pricing.

Research internship, Hitotsubashi University (Graduate School of
Economics), Tokyo — May–July 2026.
Supervisors: Prof. Toshihiro Yamada, Prof. Eiji Kurozumi.

## Motivation

Standard discretisation schemes (Euler-Maruyama) introduce a
systematic bias that only vanishes as the time grid refines. This
project implements the Beskos-Roberts method, which returns an
**unbiased** estimator at finite cost, and validates it against both
closed-form prices and the biased scheme.

## Method

1. **Lamperti transform** `Y = g(X)`, `g(x) = ∫ 1/σ` — reduces the SDE
   to unit diffusion coefficient.
2. **Girsanov-Maruyama change of measure** — removes the remaining
   drift, turning `Y` into a Brownian motion under `Q`.
3. **Poisson thinning** — the resulting path integral
   `exp(∫ c(Y_s)ds)` is expressed as the probability that a Poisson
   process has no point below `-c(Y)`, sampled by rejection.
   Intermediate values `Y_τ` are drawn from a **Brownian bridge**.
4. **Barrier payoffs** — the law of the minimum of a Brownian bridge is
   derived in closed form via the reflection principle, giving the
   crossing probability
   `q(t,s,a,b) = exp(-2/(s-t) · (g(L)-g(a))(g(L)-g(b)))`.
5. **CEV extension** — `dX_t = σX_t^β dW_t` is priced by expanding on
   infinitesimal generators: the closed-form Black-Scholes barrier
   price carries the main term and only the correction is simulated,
   acting as a control variate.

## Results

- European call under Black-Scholes: estimator matches the closed-form
  price within the confidence interval across the parameter grid.
- Barrier call: price decreasing in the barrier level `L` and
  converging to the European price as `L → 0`, as expected.
- Euler-Maruyama converges to the Beskos-Roberts price at rate
  `O(1/√N)`; the intercept of the regression in `1/√N` falls inside the
  confidence interval of the unbiased estimator.

## Usage

```bash
pip install -r requirements.txt
python src/main.py
```

## Reports

Full derivations and proofs:
- [Report 1 — Simulation of diffusion processes](reports/report1_simulation_of_diffusion_processes.pdf)
- [Report 2 — Variance reduction under the CEV model](reports/report2_variance_reduction_CEV.pdf)

## References

- Beskos, A. & Roberts, G. O. (2005). *Exact simulation of diffusions.*
  Annals of Applied Probability.

