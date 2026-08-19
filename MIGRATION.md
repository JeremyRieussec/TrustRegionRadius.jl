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
