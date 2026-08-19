# Diagnostics

Quantities derived from a completed run. Two jobs: computing the thresholds the
theory names but the solver cannot know, and doing the trace alignment once, here,
rather than in every script that reads a trajectory.

Everything on this page needs `trace = true`.

## The alignment convention

This is the one thing to get right, and it is easy to get wrong.

Entry `j` of every trajectory describes iteration `j-1`, counting from the initial
point. The **state** trajectories carry one entry more than the **per-iteration**
ones, because they also record the state the final iteration left behind:

| length `k+1` (state) | length `k` (per-iteration) |
|:--|:--|
| `:delta_trajectory`, `:grad_trajectory`, `:obj_trajectory` | `:ratio_trajectory`, `:step_trajectory` |
| `:lambda_min_trajectory`, `:tau_trajectory` | `:active_trajectory`, `:accepted_trajectory` |
| `:dist_trajectory`, `:lambda_min_true_trajectory` | `:branch_trajectory`, `:cg_iters_trajectory` |
| `:hessian_norm_trajectory` | `:rho_tilde_trajectory`, `:gamma_trajectory`, `:xi_trajectory`, `:cos_cauchy_trajectory` |

The two families agree at the **head**, so to pair a state with the iteration it
produced, drop the last entry:

```@example diagnostics
using TrustRegionRadius, ADNLPModels, NLPModels

nlp = ADNLPModel(x -> (1 - x[1])^2 + 100(x[2] - x[1]^2)^2, [-1.2, 1.0],
                 name = "Rosenbrock")
stats = tr_solve(nlp; rule = RGrad(μ = 1.0), model = ExactHessian(),
                 params = TRParams(tol = 1e-8), trace = true)

Δ = stats.solver_specific[:delta_trajectory]
s = stats.solver_specific[:step_trajectory]
(length(Δ), length(s))
```

A radius can never be shorter than the step taken inside it, and that inequality is
exactly what fails if the alignment is off by one:

```@example diagnostics
all(s[j] <= Δ[j] * (1 + 1e-10) for j in eachindex(s))
```

Aligning on the tail instead pairs `Δ_k` with `‖s_{k-1}‖` and shifts every activity,
ratio and step-versus-radius plot by one iteration. Prefer the helpers below, which
do the alignment once and correctly.

## Reading a trace

`theta_trajectory` returns `θ_k = Δ_k/‖g_k‖`, the ratio the local theory of Part II
is about. For the gradient-scaled rules it *is* the multiplier `μ_k`.

```@example diagnostics
θ = theta_trajectory(stats)
(length(θ), θ[1], θ[end])
```

`inactivity_index` is the first iteration after which the constraint never binds
again, counted from zero, or `nothing` if it was still binding at the end.
`active_fraction` reports the fraction of a tail slice on which it did bind.

```@example diagnostics
(inactivity_index(stats), active_fraction(stats; tail = 0.2))
```

`branch_counts` says which branch of the radius rule fired, and how often. On a rule
with a half-step guard, `:hold` counts the iterations where the guard refused an
expansion that the ratio alone would have allowed.

```@example diagnostics
branch_counts(stats)
```

## Thresholds

`kappa_bar` is the inactivity threshold `κ̄` of Condition C.Sg at a solution: the
criticality-anchored rules reach eventual inactivity only if their parameter exceeds
it. It is a property of the solution, so it cannot be checked before the run.

There are two conventions and they differ by a factor of two, which is the width of
the interval most parameter sweeps resolve. **Report which one produced a number.**

```@example diagnostics
(kappa_bar(nlp, [1.0, 1.0]),                                # 8/λ*, the default
 kappa_bar(nlp, [1.0, 1.0]; convention = :eigenvalue))      # 4/λ*
```

`kappa_bar_empirical` is the largest realised `‖s_k‖/‖g_k‖` along the run, so
comparing the two measures how loose the theoretical constant is:

```@example diagnostics
kappa_bar_empirical(stats)
```

## Rates and hypotheses

`observed_order` fits the local convergence order by regressing `log‖g_{k+1}‖` on
`log‖g_k‖`, restricted by default to iterations beyond `inactivity_index` and to
accepted steps. It returns `NaN` rather than a number when fewer than three usable
points remain, with `npts` reporting the actual count, so "too short" stays
distinguishable from "measured and small".

```@example diagnostics
observed_order(stats)
```

`hypotheses_report` measures the standing hypotheses of the local theory on the run
rather than assuming them. A run whose rate looks wrong should be read here first.

```@example diagnostics
hypotheses_report(stats)
```

`radius_sums` returns the three radius series of Part II. The third,
`Σ Δ_k²/M_k` with `M_k = L + max_{i≤k}‖H_i‖`, is `NaN` unless the run was traced
with `hessian_norm = true`, because there is then no recorded Hessian norm to divide
by. Absence is reported as `NaN` rather than as zero, which would read as a
converged sum.

```@example diagnostics
radius_sums(stats)
```

```@example diagnostics
withnorm = tr_solve(nlp; rule = RGrad(μ = 1.0), model = ExactHessian(),
                    params = TRParams(tol = 1e-8), hessian_norm = true, trace = true)
radius_sums(withnorm)
```

`n` is reported beside the sums so that a partial sum is never mistaken for a
converged one: on a run of forty iterations neither convergence nor divergence is
established, and the honest reading is the shape of the partial sums rather than
their value.

## API

```@docs
theta_trajectory
inactivity_index
active_fraction
branch_counts
kappa_bar
kappa_bar_empirical
observed_order
hypotheses_report
radius_sums
```
