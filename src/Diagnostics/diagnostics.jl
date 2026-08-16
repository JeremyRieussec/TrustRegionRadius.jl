# =============================================================================
# src/Diagnostics/diagnostics.jl
#
# Quantities derived from a completed run.
#
# Two jobs. The first is to compute the thresholds the theory names but the
# solver cannot know: κ̄, and the empirical constant a run actually exhibits.
# The second is to do the trace alignment once, here, rather than in every
# notebook: the state trajectories carry one entry more than the per-iteration
# ones, and pairing them by the tail rather than the head shifts every plot by
# an iteration. Nothing below asks the caller to get that right.
# =============================================================================

"""
    kappa_bar(λ_min::Real; convention = :neighbourhood) -> Float64
    kappa_bar(nlp, x_star; convention = :neighbourhood, nmax, lanczos_k) -> Float64

The inactivity threshold `κ̄` of Condition C.Sg at a solution.

`κ̄` is the constant for which `‖s_k‖ ≤ κ̄‖g_k‖` near `x*`, and the criticality-
anchored rules reach eventual inactivity only if their parameter exceeds it:
`ζ > κ̄` for [`RDFO`](@ref) and `μ_max > κ̄` for [`RGradCapped`](@ref). It is a
property of the solution, so it cannot be checked before the run; this function
computes it afterwards, which is what a threshold experiment needs in order to
say where the observed transition sits relative to the predicted one.

# The two conventions

The local lemma gives `κ̄ = 4/m` where `m` lower-bounds `∇²f` on a neighbourhood
of `x*`, and there are two ways to quote it.

- `:neighbourhood` (default) takes the concrete choice `m = λ*_min/2`, giving
  `κ̄ = 8/λ*_min`. It is what a fixed neighbourhood costs, and the extra factor
  of `2` is removable. This is the default because it is the convention Part III
  of the survey is written in throughout, and a package whose default disagrees
  with the paper it accompanies is a trap.
- `:eigenvalue` takes `m = λ_min(∇²f(x*))`, giving `κ̄ = 4/λ*_min`. The
  neighbourhood can be shrunk until `m` is as close to `λ*_min` as one likes,
  and every statement in which `κ̄` is compared with a parameter is asymptotic,
  so this is the constant that is actually attained.

Report which one a number was produced under. A threshold quoted in one
convention and compared against a parameter chosen in the other is off by a
factor of two, which is the width of the interval most sweeps resolve.

```julia
κ = kappa_bar(nlp, x_star)            # 8/λ*_min
tr_solve(nlp; rule = RDFO(ζ = 2κ))    # comfortably above the threshold
```
"""
function kappa_bar(λ_min::Real; convention::Symbol = :neighbourhood)
    convention in (:eigenvalue, :neighbourhood) || throw(ArgumentError(
        "kappa_bar: convention must be :eigenvalue or :neighbourhood, got :$convention"))
    λ_min > 0 || throw(ArgumentError(
        "kappa_bar: needs λ_min(∇²f(x*)) > 0, got $λ_min. At a singular or " *
        "indefinite critical point κ̄ is not defined and eventual inactivity is " *
        "not the question."))
    return convention === :eigenvalue ? 4 / λ_min : 8 / λ_min
end

function kappa_bar(nlp::AbstractNLPModel, x_star::AbstractVector;
                   convention::Symbol = :neighbourhood,
                   nmax::Int = 200, lanczos_k::Int = 40)
    λ = lambda_min_estimate(ExactHessian(), nlp, x_star;
                            nmax = nmax, lanczos_k = lanczos_k)
    return kappa_bar(λ; convention = convention)
end

"""
    kappa_bar_empirical(stats; inactive_only = true, tail = 1.0) -> Float64

The largest realised `‖s_k‖/‖g_k‖` along a run: the constant of Condition C.Sg
that the run actually exhibited.

`inactive_only = true` restricts to iterations on which the trust-region
constraint was inactive, where the step solves the model equation and the ratio
is the one the sharp bound `1/λ_min(H_k)` describes. On active iterations the
ratio is `Δ_k/‖g_k‖`, which says more about the rule than about the geometry.

`tail` keeps only the final fraction of the run, so that the constant is measured
where the local hypotheses hold rather than during the global phase.

Comparing this with [`kappa_bar`](@ref) is how one measures the looseness of the
theoretical constant, which Part II gives as a factor of `4` asymptotically.
"""
function kappa_bar_empirical(stats; inactive_only::Bool = true, tail::Real = 1.0)
    0 < tail <= 1 || throw(ArgumentError("kappa_bar_empirical: need 0 < tail ≤ 1"))
    s = _traj(stats, :step_trajectory)
    g = _traj(stats, :grad_trajectory)
    n = length(s)
    n == 0 && return NaN
    act = _traj(stats, :active_trajectory)
    lo = max(1, floor(Int, n * (1 - tail)) + 1)
    best = NaN
    for j in lo:n
        inactive_only && !isempty(act) && act[j] && continue
        gj = g[j]                       # ‖g_k‖ for the same k as ‖s_k‖ = s[j]
        gj > 0 || continue
        r = s[j] / gj
        best = isnan(best) ? r : max(best, r)
    end
    return best
end

"""
    theta_trajectory(stats) -> Vector{Float64}

The radius-to-criticality ratio `θ_k = Δ_k/‖g_k‖`, correctly aligned.

`θ_k > κ̄` is the sufficient condition for the trust-region constraint to be
inactive, and for the gradient-scaled rules `θ_k` is the multiplier `μ_k` itself,
so this is the sequence that the local theory of Part II is about. Length
`k+1`, one entry per iterate; pair it with the per-iteration arrays through
`θ[1:end-1]`.
"""
function theta_trajectory(stats)
    Δ = _traj(stats, :delta_trajectory)
    g = _traj(stats, :grad_trajectory)
    n = min(length(Δ), length(g))
    return [g[j] > 0 ? Δ[j] / g[j] : NaN for j in 1:n]
end

"""
    inactivity_index(stats) -> Union{Int, Nothing}

The first iteration after which the trust-region constraint never binds again,
counted from zero, or `nothing` if it was still binding at the end.

This is `k_⋆` of Part II's inactivity theorem, and it is the quantity the
mechanism is supposed to decide. Every statement about the rate belongs after
it: a rate estimated across `k_⋆` mixes the linear phase with the asymptotic one
and lands between the two.
"""
function inactivity_index(stats)
    act = _traj(stats, :active_trajectory)
    isempty(act) && return nothing
    last_active = findlast(act)
    last_active === nothing && return 0
    last_active == length(act) && return nothing
    # `act[j]` describes iteration `j-1`. If `last_active` is the final `true`,
    # the last binding iteration is `last_active - 1` and the first permanently
    # inactive one is `last_active`, counting from zero.
    return last_active
end

"""
    active_fraction(stats; tail = 0.1) -> Float64

Fraction of the last `tail` of the run on which the constraint was active.

Zero for a configuration that reaches eventual inactivity, and one for a
configuration trapped below its threshold. Part II predicts no intermediate
value in the limit, so an intermediate value is a finding.
"""
function active_fraction(stats; tail::Real = 0.1)
    0 < tail <= 1 || throw(ArgumentError("active_fraction: need 0 < tail ≤ 1"))
    act = _traj(stats, :active_trajectory)
    n = length(act)
    n == 0 && return NaN
    lo = max(1, floor(Int, n * (1 - tail)) + 1)
    return count(act[lo:n]) / (n - lo + 1)
end

"""
    radius_sums(stats; L = 0.0) -> (; sum_delta, sum_delta2, sum_delta2_over_M, n)

The three radius series of Part II, on one run.

- `sum_delta` = `Σ_k Δ_k`, which the step-driven rules make finite in the local
  regime;
- `sum_delta2` = `Σ_k Δ_k²`;
- `sum_delta2_over_M` = `Σ_k Δ_k²/M_k` with `M_k = L + max_{i≤k}‖H_i‖`, which is
  the quantity the criticality-anchored propositions actually prove finite, and
  is `NaN` unless the run was traced with `hessian_norm = true`.

The third is not a refinement of the second. `Σ Δ_k²` is equivalent to it only
while `{‖H_k‖}` is bounded, and that boundedness is exactly the hypothesis the
`M_k` form was introduced to drop.

`L` is the Lipschitz constant of the gradient. It is rarely known, and leaving it
at zero changes the constant, not whether the series converges; say which was
used when reporting a number.

Sums are over the whole trajectory including the final radius, and are reported
with `n` so that a partial sum is never mistaken for a converged one: on a run
of forty iterations neither convergence nor divergence is established, and the
honest reading is the shape of the partial sums, not their value.
"""
function radius_sums(stats; L::Real = 0.0)
    Δ = _traj(stats, :delta_trajectory)
    isempty(Δ) && return (sum_delta = NaN, sum_delta2 = NaN,
                          sum_delta2_over_M = NaN, n = 0)
    h = _traj(stats, :hessian_norm_trajectory)
    s1 = sum(Δ)
    s2 = sum(abs2, Δ)
    s3 = NaN
    if length(h) >= length(Δ)
        acc = 0.0; running = 0.0; ok = true
        for j in eachindex(Δ)
            isfinite(h[j]) || (ok = false; break)
            running = max(running, h[j])          # M_k is non-decreasing by construction
            M = L + running
            M > 0 || (ok = false; break)
            acc += Δ[j]^2 / M
        end
        ok && (s3 = acc)
    end
    return (sum_delta = s1, sum_delta2 = s2, sum_delta2_over_M = s3, n = length(Δ))
end

"""
    branch_counts(stats) -> Dict{Symbol, Int}

How many times each branch of the radius rule fired. See [`last_branch`](@ref)
for the vocabulary.

`:expand` counts the climb a gradient-scaled multiplier makes towards `κ̄`;
`:expand_capped` counts the iterations on which the user's cap refused that
climb, which is the mechanism by which a cap below `κ̄` traps the run.
"""
function branch_counts(stats)
    b = _traj(stats, :branch_trajectory)
    d = Dict{Symbol, Int}()
    for x in b
        d[x] = get(d, x, 0) + 1
    end
    return d
end

"""
    observed_order(stats; after_inactivity = true, last = 20, use = :grad)
        -> (; order, intercept, npts, from)

Local convergence order, estimated by regressing `log‖g_{k+1}‖` on `log‖g_k‖`
(or on `‖x_k − x_ref‖` with `use = :dist`, when the trace carries it).

`after_inactivity = true` restricts the regression to iterations beyond
[`inactivity_index`](@ref) and to accepted steps. This is not a refinement: a
regression that includes binding iterations mixes the linear phase with the
asymptotic one and returns an order between the two, which is an artefact.

Returns `NaN` order when fewer than three usable points remain, which is the
honest outcome on a run that reaches machine precision in nine iterations.

The claim under test is that this number depends on the model and the solver but
not on the radius mechanism, the mechanism deciding only `from`.
"""
function observed_order(stats; after_inactivity::Bool = true, last::Int = 20,
                        use::Symbol = :grad)
    key = use === :grad ? :grad_trajectory :
          use === :dist ? :dist_trajectory :
          throw(ArgumentError("observed_order: use must be :grad or :dist"))
    e = _traj(stats, key)
    acc = _traj(stats, :accepted_trajectory)
    n = length(e) - 1                       # number of iterations
    n >= 1 || return (order = NaN, intercept = NaN, npts = 0, from = nothing)

    k0 = 1
    if after_inactivity
        ki = inactivity_index(stats)
        ki === nothing && return (order = NaN, intercept = NaN, npts = 0, from = nothing)
        k0 = max(1, ki + 1)
    end
    lo = max(k0, n - last + 1)

    xs = Float64[]; ys = Float64[]
    for j in lo:n
        isempty(acc) || acc[j] || continue   # skip rejected steps
        a, b = e[j], e[j + 1]
        (a > 0 && b > 0 && isfinite(a) && isfinite(b)) || continue
        push!(xs, log(a)); push!(ys, log(b))
    end
    length(xs) >= 3 || return (order = NaN, intercept = NaN, npts = length(xs),
                               from = after_inactivity ? k0 - 1 : 0)

    x̄ = sum(xs) / length(xs); ȳ = sum(ys) / length(ys)
    sxx = sum((x - x̄)^2 for x in xs)
    sxy = sum((xs[i] - x̄) * (ys[i] - ȳ) for i in eachindex(xs))
    p = sxx > 0 ? sxy / sxx : NaN
    return (order = p, intercept = ȳ - p * x̄, npts = length(xs),
            from = after_inactivity ? k0 - 1 : 0)
end

"""
    hypotheses_report(stats) -> NamedTuple

The standing hypotheses of Part II §7, as measured on this run rather than
assumed:

- `xi_final`, `xi_max_tail`: the realised inexactness `‖H_ks_k + g_k‖/‖g_k‖`,
  which Condition C.Inx requires to tend to zero;
- `gamma_final`, `gamma_max_tail`: the Dennis–Moré residual, which the
  superlinear rate requires to tend to zero;
- `rho_final`: the acceptance ratio, which the local theory says tends to one;
- `cauchy_fraction`: the fraction of iterations on which the step was the Cauchy
  point to within `1e-12`, i.e. on which the model Hessian did not influence the
  direction.

A run whose rate looks wrong should be read here first: an order estimate is
only about the mechanism if these say the hypotheses held.
"""
function hypotheses_report(stats; tail::Real = 0.25)
    ξ = _traj(stats, :xi_trajectory)
    γ = _traj(stats, :gamma_trajectory)
    ρ = _traj(stats, :ratio_trajectory)
    c = _traj(stats, :cos_cauchy_trajectory)
    n = length(ξ)
    lo = n == 0 ? 1 : max(1, floor(Int, n * (1 - tail)) + 1)
    function tailmax(v)
        isempty(v) && return NaN
        w = @view v[min(lo, length(v)):end]
        return maximum((x for x in w if isfinite(x)); init = NaN)
    end
    function lastfinite(v)
        isempty(v) && return NaN
        i = findlast(isfinite, v)
        return i === nothing ? NaN : v[i]
    end
    return (xi_final       = lastfinite(ξ),
            xi_max_tail    = tailmax(ξ),
            gamma_final    = lastfinite(γ),
            gamma_max_tail = tailmax(γ),
            rho_final      = lastfinite(ρ),
            cauchy_fraction = isempty(c) ? NaN :
                count(x -> isfinite(x) && abs(x - 1) <= 1e-12, c) / length(c))
end

"""
    _traj(stats, key) -> Vector

A trajectory from `stats.solver_specific`, or an empty vector when the run was
not traced or the quantity was not measured. Absence is not an error here: a
diagnostic that throws on an untraced run is useless in a sweep.
"""
function _traj(stats, key::Symbol)
    ss = getfield(stats, :solver_specific)
    haskey(ss, key) || return Float64[]
    return ss[key]
end
