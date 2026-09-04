# The PreDoc logit experiment, reproduced

`task14b-predoc-reproduction.pdf` — 11 pages, 8 tables, 3 figures.

Supersedes `report/task14-predoc-logit/`, which recorded the same experiment
attempted under the constraint that nothing in the package be modified. That
report named four gaps and reported one comparison as refuted. The four gaps are
now features and the refuted comparison is made directly.

## What was added to the package

| feature | file | what it is |
|:--|:--|:--|
| `MultinomialLogit` | `src/Likelihood/models.jl` | the five-alternative logit, `M × J × p` attributes, convex |
| `SmoothedSize` | `src/Sampling/rules.jl` | the PreDoc's `Naive(a, b)` and the rising floor, as a wrapper over any rule |
| `CertifiedDecrease(variance = ...)` | `src/Sampling/rules.jl` | `:empirical` (default, unchanged) or `:outer_product` |
| `paired_op_variance` | `src/Sampling/oracles.jl` | `sᵀB̃s − (g̃ᵀs)²`, formed so it cannot come out negative |
| `FiniteSumNLP(scheme = ...)` | `src/Sampling/oracles.jl` | `:independent`, `:nested`, `:prefix` — the PreDoc's IRV, I/CRV, CRV |
| both variances on the trace | `src/Trust-region/common.jl` | `:paired_variance_trajectory` and `:paired_op_variance_trajectory` |

Every default is unchanged. `scheme` defaults to `:independent`, which is the
previous behaviour with the previous random-number consumption, and `variance`
defaults to `:empirical`.

## How it was checked

Two independent checks, and they answer different questions.

**The features, against their definitions.** 45 checks in `_audit/verify_new_features.jl`, runnable with
`julia --project=. _audit/verify_new_features.jl`.
Derivatives against central differences, the smoothing bound audited on a
300-iteration run rather than asserted, `paired_op_variance` against
`sᵀB̃s − (g̃ᵀs)²` computed the other way, and the nesting checks paired with a
control confirming that `:independent` does *not* nest. Without that control the
nesting checks would pass on a scheme that nested by accident.

**The package, against itself.** The full existing suite, 1452 tests, passes
unchanged. That is what establishes the defaults are untouched.

**The report, against the notebook.** All 16 numeric table rows were matched
against a line of the notebook's printed output, by value rather than by string,
so `1.00000` matches `1.0000e+00`. 0 unmatched.

That last check found a real problem in itself before it found none in the
report: the model-comparison row drew three of its numbers from one printed table
and one from a separate line, which no single-line matcher can verify. Rather
than carry two standing exceptions, the notebook now prints that comparison as a
row, and the check passes outright.

## What the runs say

The central claim reproduces. The common-random-variable schemes converge in 26
and 28 iterations where independent draws do not converge in 300.

Three things the iteration counts hide, all of which came out of the runs rather
than out of the PreDoc:

1. **Iterations are not the cost.** All three arms spend within 6% of the same
   number of term evaluations, because the converging runs reach the full
   population and their late iterations are expensive in proportion. The
   defensible statement is that one arm converged on the budget the other spent
   without converging.

2. **The mechanism is not confirmed.** The PreDoc attributes the effect to common
   variables keeping `f̃` from moving for reasons unrelated to the step. The
   natural diagnostic ranks the three schemes in the wrong order, and it is
   confounded anyway, because at a matched iteration index the runs are at
   different stages of convergence. The report says so and declines to draw the
   conclusion from it.

3. **The VAC hazard is real and measurable.** Counting falls in `N_k` taken while
   the *true* gradient is rising: 4 of 8 under `:prefix` against 1 of 6 under
   `:nested`, worst rise 1.78× against 1.15×. That is the ordering the PreDoc
   predicts, on runs identical in every other respect.

Two premises were measured rather than assumed. `f(0) = log 5` to 6.66e-16, and
the information identity holds at β* to 1.2% over all 100 000 individuals.

## One thing left open

The PreDoc says a mean-zero perturbation was added to β* during generation *and*
that the information identity holds. Read literally, a separate βₙ per individual
is a mixed-logit generating process fitted with a plain logit, which is
misspecification, and the identity then fails at every β. `MultinomialLogit`
implements the literal reading under `noise = σ` and the measurement is in the
report: identity error 0.064 at `noise = 0` against 1.366 at `noise = 1.5`. The
runs use `noise = 0`. The two readings are not reconciled and the report says so.

## Reproducing

```
julia --project=benchmark _audit/nb7_predoc_logit_mnl.jl
python _audit/build_saddle_notebooks.py nb7_predoc_logit_mnl.jl
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=julia-1.11 notebooks/Sampling/predoc_logit_v2.ipynb
```

About 65 seconds of solver time across 8 arms.
