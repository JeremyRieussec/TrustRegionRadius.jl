# Certification of TrustRegionRadius.jl for the Part III campaign

**Tree state.** Branch `checking-w-Claude-Code`, HEAD `38590a4` ("MLP gradient in one
batched pass, not |batch| unbatched ones"). **`git status` is clean**: nothing
uncommitted, nothing staged. Everything below was measured against that commit.

The five archives in `_Thesis_FINAL\Article3\experiments` were **not** produced from a
clean tree:

```
exp_2026-08-22_01-43-43_zeta_sweep        git_commit = 0dc242e  git_dirty = true
exp_2026-08-22_05-26-47_mu_sweep          git_commit = 0dc242e  git_dirty = true
exp_2026-08-22_10-05-30_comparison        git_commit = 0dc242e  git_dirty = true
exp_2026-08-22_12-46-51_interaction       git_commit = 0dc242e  git_dirty = true
exp_2026-08-22_20-59-14_convergence_rate  git_commit = d2aa0b3  git_dirty = true
```

Both commits are ancestors of HEAD. Between them only
`benchmark/experiments/exp7_convergence_rate.jl` changed, and between `d2aa0b3` and HEAD
only `src/Likelihood/models.jl`, so the five share the same `src/`. The irreducible
problem is the unrecorded working-tree delta: **none of the five can be reproduced as
recorded**, and no diff can recover what was in the tree at the time.

Scripts written for this audit are under `_audit/`. Nothing in `src/`, `test/`, `docs/`
or `benchmark/` was modified; nothing was staged or committed.

---

# Part A. Does the code run the rules the paper prints?

Eleven environments are defined in Appendix~B. Nine `RadiusRule` types exist.

The paper sets its own standard for what "identical" means, at
`Survey-part3-v1.tex:599-610`:

> The updates of Appendix~\ref{app: update mechanisms} are set-valued on their
> contraction and expansion branches. An implementation has to select a point from each
> interval, and the package selects the endpoint nearest the boundary the branch is
> named for, so $\Delta_{k+1} = \gamma_1\Delta_k$ on contraction, $\gamma_2\Delta_k$ on
> the intermediate branch and $\gamma_3\Delta_k$ on expansion. Every rule takes $\eta$,
> $\eta_1$ and $\eta_2$ from the solver rather than carrying its own, so a comparison at
> fixed thresholds is a comparison at genuinely fixed thresholds.

I therefore score each rule twice: whether it is a **conforming selection** from the
set-valued branch, and whether it matches that **stated selection convention**.

| paper environment | code type | branch-for-branch identical? | difference |
|:--|:--|:--|:--|
| $\Rdelta$ | `RDelta` `rules.jl:305` | **yes** | — |
| $\Rdeltastep$ | **none** | — | no type contracts on $\norm{s_k}$ on failure while using $\Delta_k$ on the other two branches |
| $\Rstep$ | `RStep` `rules.jl:371` | **yes** (at `contract_on_step = true`) | — |
| $\RAdaptstep$ | `RAdaptiveStep` `rules.jl:691` | **yes** | — |
| $\Rdfo(\omega)$ | `RDFO` `rules.jl:436` | conforming, **different selection** | code splits the paper's single contraction branch in two and returns $\gamma_2\Delta$ on the $\Delta>\zeta\omega$ half |
| $\Rgrad(\omega,\overline\mu)$ | `RGradCapped` `rules.jl:581` | **yes** | — |
| $\Rgrad(\omega)$ | `RGrad` `rules.jl:514` | **yes** | — |
| $\RAdaptiveGrad(\omega)$ | `RAdaptiveGrad` `rules.jl:753` | **NO** | the hold branch is absent |
| $\RAdaptiveGrad(\omega,\overline\mu)$ | **none** | — | no capped adaptive-multiplier type exists |
| $\Rrtr$ | `RRTR` `rules.jl:830` | **NO** | $\vartheta_k$ taken as $0$; branches on $\rho\ge\eta$, not $\rho\ge\eta_1$ |
| $\Rgrtr(\omega,\overline\mu)$ | `RRTRGrad` `rules.jl:901` | **NO** | three branches against five: no intermediate branch, no cap; branches on $\rho\ge\eta$ |

**Environments with no type: two** — $\Rdeltastep$ and $\RAdaptiveGrad(\omega,\overline\mu)$.
**Types with no environment: none.** All nine map onto an environment. `SecondOrder` and
the five `*Tau` aliases are not separate environments; they supply the $\omega$ argument.

## The four reported divergences

### 1. `RDFO` — REFUTED as stated, CONFIRMED as a selection difference

`rules.jl:436-444`, verbatim:

```julia
function update_radius!(r::RDFO, Δ::Float64, ρ::Float64, ::Bool,
                        η1::Float64, ::Float64,
                        ::Float64, crit_old::Float64, ::Float64)
    Δnew, b = ρ < η1             ? (r.γ1 * Δ, :contract) :
              Δ > r.ζ * crit_old ? (r.γ2 * Δ, :shrink)   :
                                   (r.γ3 * Δ, :expand)
```

The paper's `eqn: rdfo rule` has two branches, the first being
$[\gamma_1\Delta_k,\ \gamma_2\Delta_k]$ for $\rho_k<\eta_1$ **or** $\Delta_k>\zeta\omega_k$.

The code returns $\gamma_2\Delta$ in the second case, and $\gamma_2\Delta$ **is inside
that closed interval**. So the code is a *conforming selection*, and the claim that it
implements a different rule is refuted.

What is true is narrower and still matters: the paper says it selects
$\gamma_1\Delta_k$ **on contraction**, and the $\Delta_k>\zeta\omega_k$ case is inside
the paper's contraction branch. The code contracts by $\gamma_2 = 0.5$ there rather than
$\gamma_1 = 0.25$, so the trajectory differs by a factor of two per gated iteration even
though both are admissible.

*Experiment affected:* D6 and A4, which turn on how fast the radius falls once the gate
$\Delta>\zeta\omega$ fires. Not a correctness failure, but the two selections give
different $\zeta$ thresholds.

### 2. `RAdaptiveGrad` — CONFIRMED

`rules.jl:753-759`, verbatim:

```julia
function update_radius!(r::RAdaptiveGrad, ::Float64, ρ::Float64, ::Bool,
                        η1::Float64, ::Float64,
                        ::Float64, ::Float64, crit_new::Float64)
    R = _r_exp(ρ, η1, r.γ1, r.γ2, r.γ3, r.λ1, r.λ2)
    r.μ = max(r.μ * R, 1e-300)
    r.branch = R > 1 ? :expand : R < 1 ? :contract : :hold
```

The paper's `eqn: radaptgrad rule` is

$$\mu_{k+1} = \mu_k \text{ if } \rho_k \ge \eta_2 \text{ and } \norm{s_k}\le\tfrac12\Delta_k;
\qquad R_{\eta_1}(\rho_k)\mu_k \text{ otherwise.}$$

The code has no such branch: it multiplies by $R$ unconditionally. Its `:hold` fires only
when $R$ is exactly $1$, which is a different event. Confirmed by signature: $\Delta$,
$\eta_2$ **and** `s_norm` are all declared `::Float64` with no name, i.e. discarded —
these are exactly the three inputs the missing branch needs. Measured:

```
RAdaptiveGrad   uses: ρ, η1, crit_new    discards: Δ, accepted, η2, s_norm, crit_old
```

*Experiment affected:* every run with `RAdaptiveGrad` in the roster — A1, A2, A6, A7, A8,
and D1 if it is added there. The rule being run is not the rule the appendix prints.

### 3. `RRTRGrad` — CONFIRMED

`rules.jl:901-911`, verbatim:

```julia
function update_radius!(r::RRTRGrad, Δ::Float64, ρ̃::Float64, accepted::Bool,
                        ::Float64, ::Float64,
                        s_norm::Float64, ::Float64, crit_new::Float64)
    if !accepted || ρ̃ < r.η̃₁
        r.μ = _shrink_mu(r.μ, r.γ1); r.branch = :contract
    elseif ρ̃ >= r.η̃₂ && (!r.half_test || s_norm > 0.5 * Δ)
        r.μ *= r.γ3;                 r.branch = :expand
    else
        r.branch = :hold
    end
```

Three branches where `eqn: grtr rule` has five. Two things are missing, and the struct
proves it: `RRTRGrad` has fields `γ1, γ3, μ, μ0, η̃₁, η̃₂, half_test, Δmin, Δmax, branch`
(`rules.jl:870-880`) — **no `γ2` and no `μ_max`**.

- the intermediate branch $[\gamma_2\mu_k,\mu_k)$ for $\widetilde\rho\in[\widetilde\eta_1,\widetilde\eta_2)$
  is implemented as `:hold`, leaving $\mu$ unchanged where the paper shrinks it;
- the cap $\min\{\gamma_3\mu_k,\overline\mu\}$ is absent, so `RRTRGrad` is the uncapped
  rule and $\Rgrtr(\omega,\overline\mu)$ has no implementation at any $\overline\mu$.

The measured branch vocabulary confirms both: `RRTRGrad` emits only
`contract, expand, hold` — it can never emit `shrink`.

*Experiment affected:* none today, because `RRTRGrad` is commented out of `RULES`. It
blocks the plan's §2.3 decision "whether $\Rgrtr$ joins the roster".

### 4. `RRTR` and $\vartheta_k$ — CONFIRMED, and deliberate

`rules.jl:833-834` uses the constant `γ1`:

```julia
                  ρ̃ < 0 ? (min(r.γ2 * s_norm, r.γ1 * Δ), :contract) :
```

where the paper writes $\min\{\gamma_2\norm{s_k},\ \max[\gamma_1,\vartheta_k]\Delta_k\}$.
`grep -rn "ϑ\|vartheta\|safeguard" src/` finds no safeguard factor anywhere. But the
omission is **declared**, at `rules.jl:792`:

> The safeguard factor `θ_{k-1}` of Part I is taken as `0`, so `max[γ1, θ] = γ1`.

So this is a documented modelling choice, not an oversight. It is also a choice the
**paper cannot currently check**, because `Survey-part3-v1.tex:1735-1736` says:

> `% TO WRITE. The safeguard factors $\vartheta_k$ and $\widetilde\vartheta_{k+1}$ used
> by \Rrtr are undefined here and referred to in prose as "the safeguard factor of Part~I".`

The appendix defines neither factor, so there is nothing to implement against. **The gap
is in the paper first.**

A second `RRTR` divergence, not previously reported: the code branches on `accepted`,
which the solver sets as $\rho\ge\eta$ (`common.jl:538`), where the paper's $\Rrtr$
branches on $\rho_k<\eta_1$. When $\eta<\eta_1$ these differ, and A7 sweeps
$\eta\in\{0,0.01\}$ with $\eta_1=0.1$, so A7 is exactly the run that separates them. The
docstring at `rules.jl:794-798` argues that branching on `accepted` is *mandatory*, to
avoid leaving the radius untouched for $\rho\in[\widetilde\eta_1,\eta)$. Both readings
are defensible; the paper and the code should agree on one.

---

# Part B. Is the harness sound for a like-for-like comparison?

## B1. Constants — the rules do **not** run at fixed constants

Measured by constructing each entry of `benchmark/config.jl:28-39`
(`_audit/partB_constants.jl`):

```
rule            g1       g2       g3       Dmin       Dmax    rule-specific
RDelta          0.25     0.5      2.0      0.0        Inf
RStep           0.25     0.8      2.0      1.0e-14    Inf     contract_on_step=true
RDFO            0.25     0.5      2.0      0.0        Inf     ζ=100.0
RGrad           0.25     0.5      2.0      0.0        Inf     μ=1.0 half_test=true
RGradCapped     0.25     0.5      2.0      0.0        Inf     μ=1.0 μ_max=128.0 half_test=true
RAdaptiveStep   0.0625   0.5      4.0      1.0e-14    Inf     λ1=5.0 λ2=5.0
RAdaptiveGrad   0.0625   0.5      4.0      0.0        Inf     μ=1.0 λ1=5.0 λ2=5.0
RRTR            0.0625   0.25     2.5      1.0e-14    Inf     η̃₁=0.05 η̃₂=0.9
RRTRGrad        0.25     --       2.0      0.0        Inf     μ=1.0 η̃₁=0.05 η̃₂=0.9
```

Every previously reported difference is confirmed: `RStep` at $\gamma_2=0.8$, the Hei
pair at $\gamma_1=0.0625,\gamma_3=4.0$, `Δmin = 1e-14` on `RStep`, `RAdaptiveStep` and
`RRTR` against `0.0` on the rest, `RDFO` at $\zeta=100$.

**Not previously reported, and larger than any of them: `RRTR` and `RRTRGrad` are
commented out of `RULES`** (`benchmark/config.jl:37-38`):

```julia
    # ("RRTR",             () -> RRTR()),
    # ("RRTRGrad",         () -> RRTRGrad(μ = 1.0)),
```

The plan's roster $\mathcal{R}_8$ includes `RRTR`. As `config.jl` stands, A1, A2, A6, A7
and A8 would run seven configurations, not eight.

`RRTRGrad` also has **no $\gamma_2$ at all**, so it cannot be brought to a common
$\gamma_2$ with the others even in principle.

## B2. Thresholds ignored

Read off the method signatures (`_audit/partB_constants.jl`); an argument declared
`::Float64` with no name is discarded:

```
RDelta          uses: Δ, ρ, η1, η2                       discards: accepted, s_norm, crit_old, crit_new
RStep           uses: Δ, ρ, η1, η2, s_norm               discards: accepted, crit_old, crit_new
RDFO            uses: Δ, ρ, η1, crit_old                 discards: accepted, η2, s_norm, crit_new
RGrad           uses: Δ, ρ, η1, η2, s_norm, crit_new     discards: accepted, crit_old
RGradCapped     uses: Δ, ρ, η1, η2, s_norm, crit_new     discards: accepted, crit_old
RAdaptiveStep   uses: ρ, η1, s_norm                      discards: Δ, accepted, η2, crit_old, crit_new
RAdaptiveGrad   uses: ρ, η1, crit_new                    discards: Δ, accepted, η2, s_norm, crit_old
RRTR            uses: Δ, ρ, accepted, s_norm             discards: η1, η2, crit_old, crit_new
RRTRGrad        uses: Δ, ρ, accepted, s_norm, crit_new   discards: η1, η2, crit_old
```

All confirmed. `RRTR` and `RRTRGrad` discard **both** $\eta_1$ and $\eta_2$ and branch on
private $\widetilde\eta_1=0.05$, $\widetilde\eta_2=0.9$. `RDFO`, `RAdaptiveStep` and
`RAdaptiveGrad` discard $\eta_2$.

For `RDFO` and `RAdaptiveStep` this is **correct**: neither paper environment uses
$\eta_2$. For `RAdaptiveGrad` it is the missing hold branch of Part A.

**Consequence for A7.** A7 sweeps $\eta\in\{0,0.01\}$ at $\eta_1=0.1,\eta_2=0.9$. $\eta$
reaches the rules only through `accepted`, so:

- `RDelta`, `RStep`, `RDFO`, `RGrad`, `RGradCapped`, `RAdaptiveStep`, `RAdaptiveGrad` all
  **discard `accepted`**. For these seven, changing $\eta$ changes only *which iterates
  are taken*, not the radius branch — which is the intended reading of "acceptance is
  decoupled from scaling", and A7 measures exactly that.
- `RRTR` and `RRTRGrad` **use `accepted`**, so for them $\eta$ moves the branch as well.
  A7 therefore measures a different thing for those two than for the other seven. If
  `RRTR` is restored to the roster, A7's table mixes two meanings in one column.

## B3. `validate_thresholds`

```
Tuple{typeof(validate_thresholds), SecondOrder,   Real, Real, Real}
Tuple{typeof(validate_thresholds), RAdaptiveStep, Real, Real, Real}
Tuple{typeof(validate_thresholds), RStep,         Real, Real, Real}
Tuple{typeof(validate_thresholds), RadiusRule,    Real, Real, Real}
```

Only `RStep`, `RAdaptiveStep` and the `SecondOrder` wrapper have a method. The other
seven fall through to the `RadiusRule` no-op. In particular the appendix's standing
convention "$\RAdaptstep$ requires $\eta<\eta_1$ strictly" is enforced, and nothing
enforces the convention for `RAdaptiveGrad`, which the appendix gives the same
requirement ($0\le\eta<\eta_1$).

## B4. The activity flag — three different predicates

Subsolvers present: `EigenPoint, ExactMS, KrylovCG, KrylovCR, SteihaugCG`. (`KrylovCGLanczos`
is gone; it is now a deprecated alias for `KrylovCR`, and the deprecation fires twice in
the test suite.)

| solver | how `active` is set | predicate |
|:--|:--|:--|
| `SteihaugCG` | `_steihaug!` returns `true` only where it computed $\tau$ to land on the boundary (`subproblem.jl:215,224`), `false` on interior convergence (`:231`) and on max-iters (`:235`) | **exact geometric** |
| `ExactMS`, interior branch | `norm(sN) <= Δ && return sN, false` (`subproblem.jl:310`) | exact |
| `ExactMS`, hard case | `return Q * (sbar .+ τ .* e), true` (`:329`), $\tau$ chosen so $\norm{s}=\Delta$ | exact |
| `ExactMS`, bisection branch | `return -Q * (gq ./ (w .+ hi)), true` (`:355`) | **asserted `true`**, unconditionally, after bisecting to `sub.tol` |
| `KrylovCG`, `KrylovCR` | `_on_boundary` (`:435-438`): `hasproperty(stats,:on_boundary)` else `norm(s) >= (1 - 1e-8) * Δ` | **relative-norm test at 1e-8** |
| `EigenPoint` | forwards the inner flag; returns `true` when the eigenpoint wins, and that step is scaled `@. d *= Δ / nv` so $\norm{d}=\Delta$ (`second_order.jl:530-537`) | inner predicate, or exact |

Verified that the Krylov fallback is the one taken: `Krylov 0.10.8`, and
`:on_boundary in fieldnames(Krylov.SimpleStats)` is `false`, likewise `LanczosStats`.

**Verdict: activity is comparable within a subsolver, and not across them.** Steihaug and
the `ExactMS` hard case measure the geometry; the `ExactMS` bisection branch asserts the
answer; the Krylov wrappers apply a $10^{-8}$ relative tolerance. A run where `ExactMS`
returns a step at $\norm{s} = \Delta(1-10^{-6})$ is recorded active, and the identical
step from `KrylovCG` is recorded active too, but a step at $\Delta(1-10^{-7})$ is active
under Krylov and would be inactive under a strict test. The three claims that rest on
activity (Claims 4, 6 and the inactivity index) are safe **only if a single subsolver is
used throughout**, which the plan does satisfy: D1, D3, D6 and A1–A8 all use
`SteihaugCG`. D5 introduces `EigenPoint`, which forwards Steihaug's flag except on the
eigenpoint branch, where `true` is exact.

## B5. Branch vocabulary — not uniform

Emitted symbols, from the source and exercised in `_audit/partB_constants.jl`:

| rule | vocabulary |
|:--|:--|
| `RDelta`, `RStep` | `contract, shrink, expand` |
| `RDFO` | `contract, shrink, expand` |
| `RGrad` | `contract, shrink, expand, hold` |
| `RGradCapped` | `contract, shrink, expand, expand_capped, hold` |
| `RAdaptiveStep`, `RAdaptiveGrad` | `contract, expand, hold` (hold only when the factor is exactly 1) |
| `RRTR` | `contract, shrink, expand, hold` |
| `RRTRGrad` | `contract, expand, hold` (no `shrink`: the branch does not exist) |

`branch_counts` is therefore **not comparable across rules**. Three distinct reasons: the
Hei pair has a continuous factor and its `:hold` means "$R$ landed exactly on 1", not "the
rule declined to move"; `RGradCapped` has a fifth symbol the others cannot emit; and
`RRTRGrad` is missing `shrink` entirely. A table of branch counts across the roster would
be comparing different vocabularies.

## B6. Reset between runs

`run_matrix` (`src/Benchmark/run_matrix.jl:164-177`) deep-copies the rule, the model and
the subsolver per cell, and calls `NLPModels.reset!(nlp)` then, for sampled oracles,
`reset_sampling!(nlp)`.

`reset_sampling!` (`oracles.jl:518-530`) restores `rng`, the batches, `Ng`/`Nf`,
`samples_g`/`samples_f`/`samples_confirm`, the histories, the variances, `last_pred`,
`capped`, `H_ok` and the sampling rule. It does **not** touch `m.counters`.

Its docstring (`oracles.jl:510-516`) says:

> Restore the counters, the histories, the batches and the random stream to their
> initial state

Two readings. "The counters" may mean the *sample* counters, which it does reset; on that
reading the docstring is correct but ambiguous. On the natural reading — the NLPModels
evaluation counters — it is wrong. **In `run_matrix` the distinction does not bite**,
because `NLPModels.reset!(nlp)` at `run_matrix.jl:173` clears `m.counters` immediately
before. Standalone use of `reset_sampling!` leaves `neval_*` accumulating across runs.

---

# Part C. Correctness defects

## C1. NaN curvature classified as positive — CONFIRMED

`subproblem.jl:211` is `if dBd <= 0`, and `NaN <= 0` is `false`. Reproduction
(`_audit/partC_defects.jl`), with a model returning `NaN` for every $Bv$:

```
NaN <= 0 evaluates to: false
solve_subproblem! -> active = false, s = [NaN, NaN], cg_iters = 100
step contains NaN: true   flagged interior (active == false): true
full solve -> status = max_iter, iter = 20, solution = [-1.2, 1.0], dual_feas = 2.329e+02
```

The NaN step is produced, flagged **interior**, and the run reports `:max_iter` with the
iterate never having moved. Nothing raises and nothing in the trace says the model was
non-finite. A campaign that reads `:max_iter` as "budget exhausted" would misclassify it.

## C2. NaN ratio poisons the Hei multiplier — CONFIRMED

`rules.jl:757` is `r.μ = max(r.μ * R, 1e-300)`, and `max(NaN, x)` is `NaN`:

```
max(NaN, 1e-300) = NaN
start          mu = 1
after rho=NaN  mu = NaN   branch = hold
  recovery call 1 (rho= 0.95): mu = NaN   Delta = NaN
  ... five well-formed calls ...
mu recovers from NaN: false
```

One NaN ratio destroys the multiplier permanently. `RAdaptiveGrad` is in the roster for
A1, A2, A6, A7 and A8.

## C3. $\rho=-\infty$ is one sink — CONFIRMED, with a caveat

`common.jl:516`:

```julia
    st.ρ = (isfinite(f_cand) && st.predicted > 0) ? actual / st.predicted : T(-Inf)
```

```
(a) non-finite f_cand    status=max_iter     count(rho == -Inf) = 8 of 8
(b) non-positive pred    status=max_iter     count(rho == -Inf) = 8 of 8
```

Both record the identical value, so `:ratio_trajectory` cannot say which occurred.

**The caveat matters for the risk assessment.** Case (b) required a deliberately wrong
subsolver that returns an ascent step. Every shipped subsolver minimises the model, so
`predicted > 0` for any nonzero step, and that input to the sink is **not reachable from
`tr_solve` as configured**. In practice the sink has one live input, not three.

## C4. Only `DomainError` becomes `:exception` — CONFIRMED in principle; the cited line may be dead

`common.jl:473` is `err isa DomainError || rethrow()`. With a non-finite model Hessian and
`ExactMS`:

```
ESCAPED solve!: ArgumentError
message: ArgumentError: matrix contains Infs or NaNs
```

A non-`DomainError` escapes `solve!` entirely, so `run_matrix`'s `try` records the cell as
a failure with no stats object rather than a `:exception` status. Confirmed.

But the specific `ErrorException` at `subproblem.jl:343-346` was **not** reached: the
eigendecomposition inside `_exact_ms` raises first. That bracket loop only fails to close
when $\norm{s(\lambda)}$ stays above $\Delta$ up to $\lambda=10^{300}$, which needs a
non-finite $B$ or $g$ — and `eigen` rejects those first. **I could not construct an input
that reaches `subproblem.jl:343`; on the reading above it is unreachable.** Undetermined
whether some exotic input reaches it.

## C5. `model_grad_evals` is identically zero — CONFIRMED

`entry.jl:104` is the only method:

```julia
model_grad_evals(::AbstractTRSolver) = 0
```

`SPDTarget` calls `grad(nlp, x)` at `model_hessian.jl:346` (`phi_target`) and `:354`
(`dense_hessian`). Measured on a two-iteration `SPDTarget` run:

```
neval_grad reported        = 5
model_grad_evals(solver)   = 0   <- the correction offered
gradients the ITERATION used, at most one per accepted step + 1 = 3
```

`neval_grad - model_grad_evals` overstates the algorithm's gradient cost by the model's
own calls, here 5 against at most 3. Any evaluation-count profile that includes
`SPDTarget` charges it for gradients the iteration never asked for. D2 and D5 use
`SPDTarget`.

## C6. `true_gradient` evaluated when untraced — CONFIRMED, and the premise needs narrowing

The call sites are `finitesum.jl:79,106,112` and `expectation.jl:100,129,135`, and Julia
evaluates the argument before `sample_pre!`/`sample_post!` can discard it.

But `true_gradient` is **not** a full pass for every problem class:

```julia
true_gradient(p::PerturbedSum, x)        = grad(p.base, x)     # problems.jl:315
true_gradient(p::PerturbedExpectation, x)= grad(p.base, x)     # problems.jl:467
function true_gradient(p::FiniteSum, x)                        # problems.jl:318
function true_gradient(p::ScoredProblem, x)                    # problems.jl:522
```

For `PerturbedSum` and `PerturbedExpectation` it is $O(1)$ in $M$, and measured cost is
~1% of an untraced run at both $M=20\,000$ and $M=200\,000$. For a `ScoredProblem`, where
it really is a pass over $M$:

```
M=  20000 iter=30  untraced run 0.0544 s   31 true_gradient calls 0.0472 s  =  87% of the run
M= 200000 iter=30  untraced run 0.4595 s   31 true_gradient calls 0.4587 s  = 100% of the run
```

At $M=200\,000$ the untraced run is almost entirely work it discards. **Severe where it
applies, and it applies only to `FiniteSum` and `ScoredProblem`.** The brief's own note
holds: the deterministic path never calls it, and every experiment in the plan is
deterministic, so **this blocks no experiment**.

## C7. Model updated on accepted steps only — CONFIRMED, structural

`update_model!(c.model, c.s, c.y)` at `common.jl:559` sits inside `if st.accepted`
(`common.jl:540`). Measured on Rosenbrock with `SR1Model(mem = 5)`:

```
iterations = 101, accepted = 79, rejected = 22
update_model! calls = 79 (accepted only); rejected steps discarded = 22
```

Whether it is intended: the standard SR1 does update on rejected steps, and that is where
it earns its ability to represent negative curvature, so 22 of 101 curvature pairs are
thrown away here. But the omission is **structural, not a missing call**: on a rejected
step the code never evaluates the gradient at the trial point, so $y = g_{\text{new}} -
g_{\text{old}}$ does not exist to pass. Fixing it costs one extra gradient per rejected
iteration. *Affects A3 and A8, where SR1 is a column.*

## C8. `lambda_min_estimate` above $n=200$ — CONFIRMED

```julia
function lambda_min_estimate(model::ModelHessian, nlp, x;
                             nmax::Int = 200, lanczos_k::Int = 40,
                             vector::Bool = false)      # second_order.jl:267-269
```

`grep -rn "nmax\|lanczos_k" src/Trust-region/` returns nothing: **neither is reachable
through `tr_solve` or `TRParams`**. Measured on a separable quadratic with $n=400$ and
true $\lambda_{\min}=-2$ exactly:

```
Lanczos (n > nmax, 40 steps) = -1.999795
dense branch (nmax forced)   = -2.000000
estimate is ABOVE the truth by 0.000205  (optimistic: true)
```

A Ritz value bounds $\lambda_{\min}$ from above, so the second-order test
$\lambda_{\min}\ge-\text{tol}_H$ is optimistic: a point can be certified `:second_order`
whose true leftmost eigenvalue is more negative than $-\text{tol}_H$. On this well-spread
spectrum the error is $2\times10^{-4}$, comfortably larger than the plan's
$\text{tol}_H = 10^{-6}$. **For D5, any problem with $n>200$ can be certified
second-order on a Ritz value the user cannot tighten.** D5's designed problems are small,
so D5 as planned is safe; an aggregate second-order run over CUTEst would not be.

---

# Part D. Runs

## D1. The test suite

```
julia --project=. -e 'using Pkg; Pkg.test()'
Test Summary:        | Pass  Total     Time
TrustRegionRadius.jl | 1409   1409  8m58.7s
     Testing TrustRegionRadius tests passed
```

**1409 pass, 0 fail, 0 error.** Per-testset counts are not printed when everything passes;
the suite reports only the roll-up.

Three warnings:

```
WARNING: Main.KrylovCGLanczos is deprecated, use TrustRegionRadius.KrylovCR instead.   (x2)
┌ Warning: τ ≡ ‖g‖: this model is positive (semi)definite by construction, so λ_min(B) ≥ 0
  always and the second-order machinery is a no-op. ...
```

The first two are the test suite exercising a deprecated alias. The third is a deliberate
guard firing in a test.

## D2. What the suite does not cover

**Would it catch a regression in a rule's branching? Partly — for six rules of nine.**

`test_diagnostics.jl:31-96` pins exact branch symbols for `RDelta`, `RStep`, `RDFO`,
`RGrad`, `RGradCapped` and `RRTR`. For the other three it does not:

```julia
        # The Hei family has a continuous factor, so the branch is its position
        # relative to one.
        for r in (RAdaptiveStep(), RAdaptiveGrad(), RRTRGrad())
            update_radius!(r, 1.0, 0.95, true, η1, η2, 0.9, 1.0, 1.0)
            @test last_branch(r) in (:expand, :hold, :contract)
        end
```

A membership test over the entire vocabulary passes whichever branch fires. **A regression
that flipped expand and contract on `RAdaptiveStep`, `RAdaptiveGrad` or `RRTRGrad` would
not be caught** — and the missing hold branch of Part A.2 is in exactly that set.

`test_rules.jl` carries 8 exact-value `update_radius!` assertions, concentrated on
`RDelta`, `RStep` and `RGrad`.

**`returns_hprod` is tested nowhere.** `grep -rn "returns_hprod" test/` returns nothing.

The two weak assertions are confirmed verbatim:

```julia
test_models.jl:33      @test update_model!(m, s, y) === nothing
test_subproblem.jl:37  @test info.cg_iters >= 1      # in testset "CG takes more than one iteration"
```

The first cannot fail: `update_model!` returns `nothing` whether or not it updated
anything. The second asserts $\ge 1$ where the testset name claims $>1$; the following
line `@test !info.active` does carry content.

## D3. Smoke runs (`_audit/partD3_smoke.jl`, analytic set)

**D1 — runs.** Produces `:delta_trajectory`, $\sum\Delta$, $\sum\Delta^2$, all finite:

```
rule                     status         iter  D_end/D_0        sum D      sum D^2    finite?
RDelta                   max_iter        500          4   1.1717e+59  3.5304e+117       true
RGrad(mu0=1)             max_iter        500      0.693       347.63       241.31       true
RGradCapped(mubar=1)     max_iter        500    0.00198       6.2254       1.4063       true
RDFO(zeta=1)             max_iter        500   0.000977       7.3516       6.3547       true
```

The separation Claims 1 and 2 predict is visible: `RDelta` diverges, the anchored rules do
not.

**D3 — runs, and shows the effect.** Fraction of iterations with `cg_iters == 1` **and**
`active`:

```
config                 status         iter   frac cg1&act       cos tail
RGradCapped(mu=0.0001) max_iter      20000          1.000         1.0000
RGradCapped(mu=0.01)   first_order     483          0.004         0.4010
RGradCapped(mu=1.0)    first_order       2          0.000         1.0000
```

Complete silencing at $\overline\mu=10^{-4}$, and the run never converges.

**D4b — BLOCKED.** `PartIII-run-configs-v1.jl:231` passes $\mu_0=1$ with
$\overline\mu\in\{10^{-3},\dots,10^{2}\}$:

```
  mubar=0.001    BLOCKED: ArgumentError: RGradCapped: need μ_max ≥ μ
  mubar=0.1      BLOCKED: ArgumentError: RGradCapped: need μ_max ≥ μ
  mubar=1        first_order  iter=1379
  mubar=100      first_order  iter=20
```

Half the D4 grid cannot be constructed. Same blocker for A5, which fixes $\mu_0=1$ and
sweeps $\overline\mu$ from $10^{-3}$.

**D6 — BLOCKED at problem construction.**

```
BLOCKED: TypeError: non-boolean (SparseConnectivityTracer.HessianTracer{...}) used in boolean context
```

`sinc_1d` (`PartIII-run-configs-v1.jl:120`) branches on `abs(p[1]) < 1e-12`, and
`ADNLPModel`'s sparsity tracer cannot trace a branch on a traced value. Without the
branch the same objective builds and solves:

```
builds and solves: status = stalled, x = 10.90412165, f = -0.09132520
```

**A6 — runs.** $\mathcal{R}_8$ (with `RRTR` restored by hand) on the eight analytic
problems:

```
problem                  RDelta  RStep   RDFO   RGrad  RGradCapped  RRTR  RAdaptStep  RAdaptGrad
ROSENBR                      32     43     37      37           37    30          34          35
BEALE                         9     12      8      18           18     8           8          10
HIMMELBLAU                    8      8      9      10           10     9           9          12
POWELLSG                     21     21     28      23          448    21          21          21
WOOD                         66     88     66      90           90    62          81          81
ILLCOND                       2      2      2       2            2     2           2           2
EXTROSEN3                    32     59     34      46           46    32          34          48
TRIGQUAD                      6      6      7       6            6     5           7           6

cost matrix 8x8, finite entries 64 / 64, solved 64
performance_profile -> tau 500, prof (500, 8), prof[end,:] = [1.0 x8]
```

All eight configurations solve all eight problems and the profile builds.

**One interface mismatch found while doing this.** `run_matrix` iterates
`for (i, mk) in enumerate(problems); nlp = mk()` (`run_matrix.jl:164-165`), i.e. it wants
**bare thunks**, while `analytic_problems()` and `cutest_problems()` both return
`(name, thunk)` **tuples** (`harness.jl:75,102`). Passing either straight in raises
`MethodError: objects of type Tuple{String, ...} are not callable`. `grep -rn
"run_matrix(" benchmark/experiments/` returns **nothing** — no experiment calls it, so
the mismatch has never been exercised. A6, A4, A5 and A8 all need performance profiles.

## D4. `PartIII-run-configs-v1.jl` (`_audit/partD4_configs.jl`)

Parses. Every keyword it passes exists on the constructor it passes it to, including
`SteihaugCG(χ = 0.1, θ = 0.5, max_iters = 200)` — all three are real keywords.

```
0.2 RULES_8:            all eight construct OK
0.5 SteihaugCG(χ,θ,max_iters):  OK
1.  exp_decay():        OK
    sinc_1d():          REJECT  (TypeError, the traced branch)
2.  RGradCapped(μ=1.0, μ_max=1e-3 / 1e-2 / 1e-1):  REJECT  "need μ_max ≥ μ"
    RGradCapped(μ=1.0, μ_max=1 / 10 / 100):        OK
    RGradCapped(μ=μ_max) for μ ∈ {1e-3, 1, 1e2}:   OK      (the D3 form)
3.  RAdaptiveStep(γ2=0.5, γ3=1.2):  REJECT  "needs γ3 > 1 + γ2"   [correctly enforced]
    RAdaptiveGrad(γ2=0.5, γ3=1.2):  REJECT  "needs γ3 > 1 + γ2"   [correctly enforced]
```

**`RGradCapped` does not accept $\overline\mu<\mu_0$**, which A5 and D4b both need. The
Hei requirement $\gamma_3>1+\gamma_2$ **is** enforced, and the file's
$\gamma_2=0.5,\gamma_3=2.0$ satisfies it.

## D5. CUTEst

```
[ Info: CUTEst: 185 problems selected
cutest_problems(min_var=2, max_var=200, max_con=0) -> 185 problems
expected 185; difference = +0
```

**Installed, and the count is exactly 185.** No change to Section 5.

---

# Part E. The certificate

- **Fit after the fixes listed below.** Not fit as is.
- The rules the paper prints and the rules the code runs **differ in five places**:
  `RAdaptiveGrad` (missing hold branch), `RRTR` ($\vartheta_k\equiv0$, and branches on
  $\rho\ge\eta$ not $\rho\ge\eta_1$), `RRTRGrad` (no intermediate branch, no cap),
  $\Rdeltastep$ (no implementation), $\RAdaptiveGrad(\omega,\overline\mu)$ (no
  implementation). `RDFO` differs in *selection* but conforms to the set-valued rule.
- A comparison at fixed constants **is not** achievable with the current
  `benchmark/config.jl`: $\gamma_2$ differs by rule, the Hei pair runs at different
  $\gamma_1,\gamma_3$, `Δmin` differs, `RDFO` runs at $\zeta=100$, `RRTR` and `RRTRGrad`
  are commented out, and `RRTRGrad` has no $\gamma_2$ to harmonise.
- The activity indicator **is not** comparable across subsolvers, but **is** comparable
  within one, and every experiment in the plan except D5 uses `SteihaugCG` throughout, so
  the plan as written is safe.
- The test suite **would not** catch a regression in a radius rule's branching for
  `RAdaptiveStep`, `RAdaptiveGrad` or `RRTRGrad`; it **would** for the other six.
- Experiments **D1, D3, A1, A2, A4, A6 can be run today** (A4 and A6 need the `run_matrix`
  thunk shape and, for the full roster, `RRTR` uncommented). **D4b, D6 and A5 cannot.**
  Blockers: D4b and A5 need `RGradCapped` to accept $\overline\mu<\mu_0$; D6 needs
  `sinc_1d` to build.

## Fix list, ordered by experiments unblocked

 $$
  \Delta_{k+1} = 
  \begin{cases}
      \gamma_1 \|s_k\| & \text{if } \rho_k < \eta_1, \\
      \gamma_2 \Delta_k, \Delta_k) & \text{if } \rho_k \in [\eta_1, \eta_2), \\
      \max(\Delta_k, \gamma_3 \|s_k\|) & \text{if } \rho_k \geq \eta_2,
  \end{cases}
$$
where $0 < \gamma_1 \leq \gamma_2 \leq 1 < \gamma_3.$


| # | file:line | change | unblocks |
|--:|:--|:--|:--|
| 0 | implement `Rdeltastep` | use definition presented before table  | **D1, D3, A1, A2, A4, A6** |
| 1 | `src/Radius_updates/rules.jl:569` | relax `μ_max >= μ` to allow $\overline\mu<\mu_0$, clamping $\mu_0\leftarrow\min(\mu_0,\overline\mu)$ at construction, or let the plan set $\mu_0=\overline\mu$ | **D4b, A5** (and D3's grid at small $\overline\mu$) |
| 2 | `PartIII-run-configs-v1.jl:120` | remove the traced branch from `sinc_1d`; `sin(x)/x` builds and solves without it, and the guard is unreachable from $x_0=10.95$ | **D6, A9** |
| 3 | `src/Radius_updates/rules.jl:870-911` | add `γ2` and `μ_max` to `RRTRGrad` and the two missing branches | the §2.3 decision on whether $\Rgrtr$ joins the roster |
| 4 | `src/Radius_updates/rules.jl:753-759` | add the hold branch: `ρ ≥ η2 && s_norm ≤ 0.5Δ` leaves `μ` unchanged; the signature must stop discarding `Δ`, `η2`, `s_norm` | **A1, A2, A6, A7, A8** — makes the rule the one the appendix prints |
| 5 | `benchmark/config.jl:37-38` | uncomment `RRTR` and `RRTRGrad`, and harmonise $\gamma_1,\gamma_2,\gamma_3,\Delta_{\min}$ across all of `RULES` except those adaptive (Hei style) because the multiplative factor varies with respect to the ratio, so $\gamma_1,\gamma_2,\gamma_3$ are used to know the range in which the updates operate | **A1, A2, A6, A7, A8** (roster $\mathcal{R}_8$ is eight, not seven) |
| 6 | `src/Benchmark/run_matrix.jl:164` or `harness.jl:75,102` | keep the `run_matrix` structure but I will continue using `harness.jl` for the actual calls | **A4, A5, A6, A8** |
| 7 | `src/Radius_updates/rules.jl:757` | guard the NaN: `isfinite(R) || return` before `r.μ = max(r.μ * R, 1e-300)` | any run with `RAdaptiveGrad` — A1, A2, A6, A7, A8 |
| 8 | `src/Subproblem/subproblem.jl:211,220,231` | test `!(dBd > 0)` rather than `dBd <= 0`, so `NaN` takes the negative-curvature branch | all; converts a silent `:max_iter` into a visible failure |
| 9 | `src/Trust-region/common.jl:473` | catch the exception types the subsolvers actually raise, or catch broadly and record `:exception` | all; a bracket failure currently aborts the whole matrix cell |
| 10 | `src/Trust-region/entry.jl:104` | give `SPDTarget` a real `model_grad_evals`, or drop the correction and say so | D2, D5 evaluation counts |
| 11 | `test/test_diagnostics.jl:93-96` | replace the membership assertion with exact branches for the Hei trio | protects fixes 4 and 9 from regressing |
| 12 | `src/Second_order/second_order.jl:267` | thread `nmax`/`lanczos_k` through `TRParams` | D5 at $n>200$ |

Item 1 is the single highest-value change: it alone unblocks two experiments.

---

# Anything I found that is not in this brief

1. **`RRTR` and `RRTRGrad` are commented out of `benchmark/config.jl:37-38`.** The plan's
   roster $\mathcal{R}_8$ names `RRTR`. Every aggregate run as configured today would
   produce seven columns where the paper's table has eight, and nobody reading the output
   would see a missing row.

2. **No experiment calls `run_matrix`.** `grep -rn "run_matrix(" benchmark/experiments/`
   is empty. The function that produces the performance profiles at the centre of
   Section 5 is exercised only by tests and docs, and its input shape disagrees with what
   `analytic_problems()` and `cutest_problems()` return. The first real use will hit that.

3. **`RRTRGrad` has no `γ2` field at all**, so it cannot be brought to common constants
   with the rest of the roster even in principle. This is a stronger statement than "no
   intermediate branch": there is no parameter to set.

4. **`RRTR` and `RRTRGrad` branch on `accepted`, i.e. on $\rho\ge\eta$, where the paper
   branches on $\rho\ge\eta_1$.** A7 sweeps $\eta$ at fixed $\eta_1$, so A7 is precisely
   the experiment that separates the two readings, and its column for `RRTR` would not
   mean what its columns for the other seven mean. `rules.jl:794-798` argues the code's
   choice is mandatory for weak admissibility. The paper should adopt it or the code
   should change; they cannot both stand.

5. **The paper's $\vartheta_k$ gap is a paper gap first.** `Survey-part3-v1.tex:1735-1736`
   carries `% TO WRITE. The safeguard factors ... are undefined here`. There is nothing
   for the code to implement, and `rules.jl:792` documents taking $\vartheta\equiv0$.
   Reporting this as a code defect inverts the dependency.

6. **`true_gradient` is $O(1)$ for `PerturbedSum` and `PerturbedExpectation`**, not a full
   pass. The eager-evaluation cost is real but only on `FiniteSum` and `ScoredProblem`,
   where it is severe (87–100% of an untraced run). Any estimate of the campaign's
   sampled-run cost based on "a full pass per iteration on every sampled problem" is
   wrong in both directions depending on the class.

7. **The `ExactMS` `ErrorException` at `subproblem.jl:343-346` appears unreachable.** The
   eigendecomposition earlier in `_exact_ms` rejects the only inputs that would stop the
   bracket closing. The defensive branch is sound; it is also, as far as I could
   construct, dead.

8. **$\rho=-\infty$ has one live input, not three.** A correct subsolver cannot produce
   `predicted ≤ 0` for a nonzero step, so only the non-finite-$f$ path reaches the sink
   from `tr_solve`. The trace ambiguity is real and narrower than reported.

9. **The paper's own $\lambda^*$ formula for the sinc family checks out exactly.**
   $1/\sqrt{1+x_k^2}$ against $f''(x_k)$ by central differences at $k=1,3,5$ agrees to ten
   digits, and $\lambda^*=0.0913252028$ at $k=3$ matches the $0.09133$ printed in the
   older Part III draft. `sinc_thresholds` is sound; only the `ADNLPModel` wrapper is not.

10. **The suite emits a deprecation twice**: `KrylovCGLanczos` is now an alias for
    `KrylovCR`. Worth noting because the brief's Part B.4 asks about `KrylovCR` and an
    earlier audit of this package reported a live defect in `KrylovCGLanczos` (it passed a
    `radius` keyword `Krylov.cg_lanczos` has never accepted). Verified fixed:
    `Krylov.cr` does accept `radius`, `Krylov.cg_lanczos` still does not, and the package
    no longer calls the latter.

11. **`SteihaugCG` burns its full iteration budget on a NaN model** (100 iterations in C1)
    rather than stopping. Combined with C1 this makes a non-finite model both silent and
    slow.
