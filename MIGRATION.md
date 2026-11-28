# Migration

What to edit in code written against an earlier version of this package. Every
claim below was checked by running it against the current code, and each entry
says whether the old spelling **throws** (you will be told) or is **silently
reinterpreted** (you will not).

## 1. The solver split by problem class

`TRSolver` no longer exists. There are three solvers, one per problem class, and
`tr_solve` picks the right one from the model you hand it.

| before | now |
|---|---|
| `TRSolver(nlp; ...)` on a deterministic problem | `DeterministicTRSolver(nlp; ...)` |
| `TRSolver(nlp; ...)` on an expectation | `ExpectationTRSolver(nlp; ...)` |
| `TRSolver(nlp; ...)` on a finite sum | `FiniteSumTRSolver(nlp; ...)` |

**Throws.** `TRSolver` is not defined, so an old call raises `UndefVarError`.

`tr_solve(nlp; ...)` is unchanged and dispatches for you. Prefer it unless you
need the solver object itself.

## 2. `SampledNLP` became an abstract supertype

`SampledNLP` is still exported, but it is now abstract, with subtypes
`ExpectationNLP` and `FiniteSumNLP`. The distinction is not cosmetic: a finite
sum has a population `M`, so `N_k` can reach it and the step become exact, and
`true_stop = :full` can be answered. An expectation has neither.

- Code that **dispatched** on `::SampledNLP` still works and still means
  "either sampled class".
- Code that **constructed** `SampledNLP(prob, rule)` must name the class it
  meant. **Throws** — an abstract type has no constructor.

`LikelihoodNLP` is an alias for `FullBatchNLP` (`LikelihoodNLP === FullBatchNLP`
is `true`), so that rename needs no edit.

## 3. `TRParams` keywords are ASCII, with the subscript spellings kept

`η1`, `η2`, `Δ0` are the canonical names. `η₁`, `η₂`, `Δ₀` work as keywords
**and** as property names, so nothing needs editing:

```julia
TRParams(η₁ = 0.2).η1 == 0.2      # true
TRParams(η1 = 0.2).η₁ == 0.2      # true
```

`η` defaults to `η1`, so existing calls that never set `η` are unaffected.

## 4. Acceptance decoupled from scaling, and the factors renumbered

`update_radius!` takes an `accepted::Bool` third argument, and every rule now
obeys `0 < γ1 ≤ γ2 < 1 < γ3`: `γ1` and `γ2` contract, `γ3` expands. Four rules
had their factors renumbered.

| rule | before | now |
|---|---|---|
| `RGrad` | `γ₁ = 0.25`, `γ₂ = 2.0` | `γ₁ = 0.25`, `γ₂ = 0.5`, `γ₃ = 2.0` |
| `RGradCapped` | `γ₁ = 0.25`, `γ₂ = 2.0` | `γ₁ = 0.25`, `γ₂ = 0.5`, `γ₃ = 2.0` |
| `RRTR` | `γ₀ = 0.0625`, `γ₁ = 0.25`, `γ₂ = 2.5` | `γ₁ = 0.0625`, `γ₂ = 0.25`, `γ₃ = 2.5` |
| `RRTRGrad` | `γ₁ = 0.25`, `γ₂ = 2.0` | `γ₁ = 0.25`, `γ₃ = 2.0` |

### Keyword calls: throws

`check_factors` rejects an expansion factor in a contraction slot, so an old
keyword call cannot be silently reinterpreted:

```julia
RGrad(γ1 = 0.25, γ2 = 2.0)
# ArgumentError: RGrad: need 0 < γ2 < 1, got γ2 = 2.0
```

### Positional calls: the case that can pass silently

Four rules take positional factors — `RDelta(γ1, γ2, γ3)`, `RStep(γ1, γ2, γ3)`,
`RDFO(γ1, γ2, γ3, ζ)`, `RGrad(γ1, γ2, γ3, μ)`. The renumbering **moved what
slot 3 means**. A positional call is checked only against the ordering, so one
whose numbers happen to satisfy `0 < γ1 ≤ γ2 < 1 < γ3` is accepted with its
arguments reassigned:

```julia
r = RGrad(0.0625, 0.25, 2.5, 1.0)   # accepted
r.γ1, r.γ2, r.γ3                    # (0.0625, 0.25, 2.5)
```

Nothing warns, because nothing can tell that from a deliberate call. **Search
your code for positional rule constructors and convert them to keywords**, which
are order-independent and therefore cannot be reinterpreted:

```julia
RGrad(; γ1 = 0.25, γ2 = 0.5, γ3 = 2.0, μ = 1.0)
```

Old calls that put an expansion factor in slot 2 still throw, so only the
ordering-compatible ones are at risk.

## 5. `Δmin` and `Δmax` apply on every branch

Every rule carries both and applies them on **every** branch. Previously `Δmax`
was honoured only on the expansion branch of some rules and absent from others.
`check_bounds` enforces `0 ≤ Δmin ≤ Δmax` at construction. A rule that relied on
`Δmax` being ignored on contraction will now clamp where it did not before —
this changes results rather than raising, so re-run rather than assuming.

## 6. `KrylovCGLanczos` is now `KrylovCR`

The wrapper was generated from `Krylov.cg_lanczos`, which has no `radius`
keyword, so every call raised a `MethodError` — it had never run. It is now
generated from `Krylov.cr`, which accepts `radius` and handles indefinite `B`.
`KrylovCGLanczos` remains as a deprecated alias, so no edit is required.

## 7. `model_hessian_norm`: `power_iters` is now `lanczos_k`

The large-`n` branch is a Lanczos iteration rather than a power iteration on
`H²`, and the keyword names the number of Lanczos steps. **Throws** — an old
`power_iters = 300` is an unsupported keyword. The default `lanczos_k = 40`
costs fewer matrix-vector products than the thirty power steps it replaced and
is more accurate; see the docstring for the measurements.

## 8. `true_stop = :test` and `:both` read their bound on the largest batch

The confidence bound of `:test` and `:both` is now read only on the largest batch
the oracle allows, unless the new `TRParams` keyword `stop_N_min` names a smaller
one. The largest batch is the smallest of the rule's `N_max`, the oracle's
`budget` and, on a finite sum, the population. Before, the bound was read on every
batch, and a batch of two whose scores had all vanished could certify
`:first_order` at a point that was not critical.

**Silently reinterpreted.** A sampled run under `:test` or `:both` that used to
stop on a smaller batch no longer stops there. `:test` refuses, `:both` takes its
full pass, and section 9 describes what the run does next. `TRParams(stop_N_min = 1)`
reads the bound on every batch, as before, except a batch on which the step has just
stalled. Nor is the bound read any longer on the initial batch, whose variance has not
been estimated yet, unless that batch is the population. A minimum between the two
gives no guarantee. With `stop_N_min = 30`, 4 of 12 probit runs whose rule never draws fewer
than 32 observations still stopped wrongly on batches of 32. The setting is
described under `TRParams`.

## 9. A confirming mode moves the run to the largest batch

Under `:test`, `:both` and `:full`, three things change on a sampled run.

- A refused stop raises a floor under the sample size, at the largest batch, for the
  rest of the run, whatever the sampling rule asks for. `FixedSample` and
  `GeometricSample` are no exception. `TRParams(on_refusal = :continue)` keeps the
  earlier behaviour.
- A stall on a smaller batch no longer ends the run. The iteration is kept as a
  rejected step, traced and handed to the callback, and the run moves to the largest
  batch. No keyword turns this off.
- A stall on the largest batch still ends the run, but its batch is first offered to
  the stopping test, and the run reports `:first_order` (`:second_order` under
  `tol_H`) when it passes. A stalled batch is judged only exactly, when it is the
  population or by a full pass, so under `:test` a stall on a largest batch that is a
  sample still ends `:stalled`.

**Silently reinterpreted.** Statuses, iteration counts and sample counts of these runs
change. On a logistic regression and a probit, 24 runs that gave 18 correct stops, 5
stalls on batches of two and one `max_iter` now give 24 correct stops in every mode,
and `:full` draws 3 926 976 samples against 19 934 099. On an expectation whose
largest batch cannot resolve `tol`, a `:test` run that used to reach `max_iter`
cheaply now spends its remaining iterations on that batch. `TRParams` gives the
numbers. A traced run under a confirming mode carries a new key, `:sample_floor_from`,
the first iteration drawn under the floor.
`:sample_cap_hits` still counts only the iterations at which the rule itself asked for
the cap.

**Throws.** `FiniteSumTRSolver` refuses a confirming mode on an oracle with
`scheme = :nested` and `replace = true` whose largest batch is the population,
because a batch of `M` drawn that way repeats some observations and misses others.
Pass `replace = false`, or use `:independent` or `:prefix`. `ExpectationTRSolver`
refuses a confirming mode on an oracle whose budget is `typemax(Int)` and whose rule
has no `N_max`, since there is then no largest batch to move to.

Runs under `:none`, and every deterministic run, are unchanged. We compared 91 of
them, traces included, bit for bit against the previous version.
