# =============================================================================
# benchmark/experiments/exp16_sinc_dichotomy.jl
#
# EXPERIMENT 16 -- the activity dichotomy on sin(x)/x, and the boundedness of
# the multiplier.
#
# WHAT THIS ESTABLISHES, AND WHAT IT DOES NOT
#
# Part II proves half of the dichotomy. Below the threshold `thm: sinc rgrad`
# gives an active constraint at every large k. Above it, inactivity arrives from
# the other direction through θ_k > κ̄ with κ̄ = 8/λ*_j. On the interval
# θ ∈ [1/λ*_j, 8/λ*_j) neither theorem speaks, while the exact criterion of
# `eqn: sinc activity` decides the question outright, the problem being
# one-dimensional.
#
# This run does not confirm a proved dichotomy. It locates the true constant
# inside a gap where the analysis has a factor of 8 to spare.
#
# NUMBERING. The prompt for this task asked for exp14. exp14 and exp15 were
# already taken by exp14_acceptance_band.jl and exp15_slow_saddle.jl, and
# renumbering an existing experiment is forbidden, so this is exp16.
#
# THE MODEL. `sinc_1d` of PartIII-run-configs-v1.jl:120 carries a guard
# `abs(p[1]) < 1e-12 ? 1.0 : sin(p[1])/p[1]`. That file is on neither repository,
# and the guarded form is rejected by the current ADNLPModels:
# SparseConnectivityTracer cannot take a data-dependent branch. The model is
# therefore defined here, unbranched. The branch is unreachable on this
# experiment in any case, since the iterates sit near y_j in [4.49, 23.52].
#
# THE CRITICAL POINTS come from task 1, read at run time from
# `_Thesis_FINAL/Article3/sinc_experiment/data/roots.dat`, which carries them at
# 240 bits. They are not recomputed here.
#
# THE SETTINGS. Section `subsec: settings` of Survey-part3-v1.tex says every
# number in the paper is fixed there and carries a TODO in place of the table,
# so there is nothing to check config.jl against. The sole source is
# benchmark/config.jl: η = 0.1, η1 = 0.1, η2 = 0.9, γ1, γ2, γ3 = 0.25, 0.5, 2.0.
#
#   julia --project=benchmark -e 'include("benchmark/initialisation.jl"); sinc_dichotomy()'
#
# No `using` and no `include` here. initialisation.jl loads the packages,
# harness.jl, archive.jl and config.jl once, in order.
# =============================================================================

# -----------------------------------------------------------------------------
# The critical points, from task 1
# -----------------------------------------------------------------------------

"Default location of task 1's output. Override with TRR_SINC_DATA."
sinc_data_dir() = get(ENV, "TRR_SINC_DATA",
    normpath(joinpath(@__DIR__, "..", "..", "..", "_Thesis_FINAL", "Article3",
                      "sinc_experiment", "data")))

"""
    sinc_roots() -> Dict{Int, NamedTuple}

`y_j` and `λ*_j = f''(y_j)` for the minimisers, read from task 1's `roots.dat`.

Task 1 computed these at 240 bits and is the single source. We do not recompute
them and we do not use the Newton routine of `_audit/partD_sinc_cutest.jl`.
"""
function sinc_roots()
    path = joinpath(sinc_data_dir(), "roots.dat")
    isfile(path) || error("exp16 needs task 1's output. No roots.dat at $path. " *
                          "Set TRR_SINC_DATA to the directory that holds it.")
    out = Dict{Int, NamedTuple}()
    for line in eachline(path)
        startswith(strip(line), "#") && continue
        p = split(strip(line))
        length(p) >= 5 || continue
        p[5] == "minimiser" || continue
        j = parse(Int, p[1])
        out[j] = (y = parse(Float64, p[2]), λ = parse(Float64, p[3]))
    end
    isempty(out) && error("roots.dat at $path held no minimisers")
    return out
end

const SINC_J = [1, 3, 5, 7]

"The offset of the existing table's start point: 10.954122 = y_3 + 0.05."
const SINC_OFFSET = 0.05

"f(x) = sin(x)/x, one-dimensional. Unbranched, for the reason in the header."
sinc_model(x0::Real) = ADNLPModel(p -> sin(p[1]) / p[1], [float(x0)], name = "SINC1D")

sinc_model(x0::BigFloat) =
    ADNLPModel(p -> sin(p[1]) / p[1], [x0], name = "SINC1D-BF")

"f''(x) in closed form, for the activity criterion and the hazard-1 check."
sinc_curv(x) = (2sin(x) - 2x * cos(x) - x^2 * sin(x)) / x^3

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
# Two stopping rules, because the two arms are not comparable. Below the
# threshold the run converges linearly and does not terminate, so it stops on
# the budget or on the rounding level of f. Above it the run finishes in a
# handful of iterations. Their iteration counts never share a column.
const SD_KMAX_BELOW = 4_000
const SD_KMAX_ABOVE = 200
const SD_TOL        = 1e-12

sd_params(; kmax = SD_KMAX_BELOW, tol = SD_TOL, Δ0 = 1.0) =
    TRParams(η = SOLVER_PARAMS.η, η1 = SOLVER_PARAMS.η1, η2 = SOLVER_PARAMS.η2,
             Δ0 = Δ0, Δmin = 0.0, Δmax = Inf, tol = tol, max_iterations = kmax)

sd_params_big(; kmax = SD_KMAX_BELOW, tol = BigFloat("1e-40"), Δ0 = 1.0) =
    TRParams{BigFloat}(η = SOLVER_PARAMS.η, η1 = SOLVER_PARAMS.η1,
                       η2 = SOLVER_PARAMS.η2, Δ0 = Δ0, Δmin = 0.0, Δmax = Inf,
                       tol = tol, max_iterations = kmax)

# -----------------------------------------------------------------------------
# One run
# -----------------------------------------------------------------------------

"""
    sd_run(rule, j, roots; ...) -> NamedTuple

One `tr_solve` on the sinc problem started at `y_j + 0.05`.

`ExactMS` in `Float64` and `SteihaugCG` in `BigFloat`. The two coincide in one
dimension, since conjugate gradient's first direction is `-g` and the exact
minimiser along it is `-g/a`, and we verify that agreement rather than assume
it. `ExactMS` cannot run in `BigFloat`, because it calls `eigen!` and LAPACK has
no method for `Symmetric{BigFloat}`.
"""
function sd_run(rule, j, roots; big = false, kmax = SD_KMAX_BELOW, Δ0 = 1.0,
                tol = nothing)
    y = roots[j].y
    x0 = big ? BigFloat(string(y)) + BigFloat("0.05") : y + SINC_OFFSET
    nlp = sinc_model(x0)
    xs = Any[copy(nlp.meta.x0)]
    p = big ? sd_params_big(kmax = kmax, Δ0 = Δ0,
                            tol = tol === nothing ? BigFloat("1e-40") : tol) :
              sd_params(kmax = kmax, Δ0 = Δ0,
                        tol = tol === nothing ? SD_TOL : tol)
    t0 = time()
    st = tr_solve(nlp; rule = rule, model = ExactHessian(),
                  subsolver = big ? SteihaugCG() : ExactMS(),
                  params = p, trace = true,
                  callback = (_n, _s, s) -> push!(xs, copy(s.solution)))
    dt = time() - t0

    ss = st.solver_specific
    Δ  = get(ss, :delta_trajectory,  Float64[])
    g  = get(ss, :grad_trajectory,   Float64[])
    sn = get(ss, :step_trajectory,   Float64[])
    ac = get(ss, :active_trajectory, Bool[])
    θ  = theta_trajectory(st)

    # theta_k * a_k, the scale-free activity criterion of `eqn: sinc activity`.
    a  = [Float64(sinc_curv(Float64(x[1]))) for x in xs]
    n  = min(length(θ), length(a))
    θa = [Float64(θ[i]) * a[i] for i in 1:n]

    # Hazard 1. The closed-form one-dimensional step, checked every iteration.
    worst = 0.0; nneg = 0
    for k in eachindex(sn)
        k <= length(xs) || break
        xk = Float64(xs[k][1]); ak = sinc_curv(xk); gk = Float64(g[k]); Δk = Float64(Δ[k])
        ak <= 0 && (nneg += 1)
        pred = (ak > 0 && abs(gk) / ak <= Δk) ? abs(gk) / ak : Δk
        worst = max(worst, abs(pred - Float64(sn[k])) / max(pred, 1e-300))
    end

    # Hazard 2. The limit point is the nearest critical point, never assumed.
    xend = Float64(st.solution[1])
    nearest_j, nearest_d = 0, Inf
    for (jj, r) in roots
        d = abs(xend - r.y); d < nearest_d && (nearest_d = d; nearest_j = jj)
    end

    return (j = j, status = st.status, iter = st.iter, time = dt,
            gnorm = Float64(st.dual_feas), x_end = xend,
            limit_j = nearest_j, limit_dist = nearest_d,
            inactivity = inactivity_index(st),
            theta_a = θa, theta = Float64.(θ), active = ac,
            accepted = count(get(ss, :accepted_trajectory, Bool[])),
            rejected = st.iter - count(get(ss, :accepted_trajectory, Bool[])),
            branches = branch_counts(st),
            kbar_emp = kappa_bar_empirical(st),
            kbar_emp_all = kappa_bar_empirical(st; inactive_only = false),
            step_err = worst, n_negcurv = nneg,
            delta = Float64.(Δ), grad = Float64.(g),
            precision = big ? "BigFloat" : "Float64")
end

# -----------------------------------------------------------------------------
# Three classes, not two
# -----------------------------------------------------------------------------

"""
    sd_class(r; window) -> (Symbol, Int, Int)

`:inactive`, `:active` or `:neither`, with the tail window and the number of
accepted iterations in it.

A run that alternates to the end is `:neither`. The experiment has to be able to
falsify the dichotomy rather than assume it, so the third class exists and is
reported wherever it occurs.
"""
function sd_class(r; window::Int = 0)
    n = length(r.active)
    n == 0 && return (:undetermined, 0, 0)
    w = window > 0 ? min(window, n) : max(1, min(n, cld(n, 4)))
    tail = r.active[(n - w + 1):n]
    nacc = w
    r.inactivity !== nothing && return (:inactive, w, nacc)
    all(tail) && return (:active, w, nacc)
    return (:neither, w, nacc)
end

"Whether the run classifies as eventually active. The bisection predicate."
sd_is_active(r) = sd_class(r)[1] === :active

# -----------------------------------------------------------------------------
# Bisection
# -----------------------------------------------------------------------------

"""
    sd_bisect(mkrule, j, roots, lo, hi; ...) -> NamedTuple

Locate the transition to three significant figures between `lo`, which must
classify as eventually active, and `hi`, which must not.

The predicate is checked at the brackets first. A bisection on a predicate that
does not hold at its brackets returns a number with no meaning, so we report the
failure instead of a value.
"""
function sd_bisect(mkrule, j, roots, lo, hi; kmax = SD_KMAX_BELOW, Δ0 = 1.0,
                   digits = 3, maxit = 40)
    rlo = sd_run(mkrule(lo), j, roots; kmax = kmax, Δ0 = Δ0)
    rhi = sd_run(mkrule(hi), j, roots; kmax = kmax, Δ0 = Δ0)
    alo, ahi = sd_is_active(rlo), sd_is_active(rhi)
    (alo && !ahi) || return (ok = false, star = NaN, lo = lo, hi = hi,
                             cls_lo = sd_class(rlo)[1], cls_hi = sd_class(rhi)[1],
                             iters = 0)
    a, b = float(lo), float(hi)
    it = 0
    while it < maxit && (b - a) > abs(b) * 10.0^(-digits - 1)
        m = sqrt(a * b)                      # geometric, the scale is logarithmic
        sd_is_active(sd_run(mkrule(m), j, roots; kmax = kmax, Δ0 = Δ0)) ?
            (a = m) : (b = m)
        it += 1
    end
    return (ok = true, star = sqrt(a * b), lo = a, hi = b,
            cls_lo = :active, cls_hi = :inactive, iters = it)
end

"""
    sd_scan(mkrule, j, roots, lo, hi; n) -> (vals, classes)

The classification on a logarithmic grid across the whole bracket.

A coarse grid can straddle a window and report a monotone profile that is not
one, which is what an earlier version of this file did for `RDFO`. The grid
therefore spans the same range as the bisection and is fine enough to resolve a
window.
"""
function sd_scan(mkrule, j, roots, lo, hi; n = 24, kmax = SD_KMAX_BELOW, Δ0 = 1.0)
    vals = exp10.(range(log10(lo), log10(hi); length = n))
    cls  = [sd_class(sd_run(mkrule(v), j, roots; kmax = kmax, Δ0 = Δ0))[1]
            for v in vals]
    return (vals, cls)
end

"""
    sd_monotone(cls) -> Bool

Whether a classification profile is monotone in the parameter, meaning every
`:active` precedes every non-`:active`. A bisection on a non-monotone predicate
returns a number with no meaning, so the verdict is reported beside every
threshold and a non-monotone profile is reported as a window instead.
"""
function sd_monotone(cls)
    seen_non = false
    for c in cls
        c === :active ? (seen_non && return false) : (seen_non = true)
    end
    return true
end

"The first index at which the profile leaves `:active`, and the last before it returns."
function sd_window(vals, cls)
    i = findfirst(c -> c !== :active, cls)
    i === nothing && return (nothing, nothing)
    j = findlast(c -> c !== :active, cls)
    return (vals[i], vals[j])
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function sinc_dichotomy()
    roots = sinc_roots()
    arch  = ExperimentArchive(tag = "sinc_dichotomy")
    @info "exp16: minimisers $(SINC_J), y_j and lambda*_j read from task 1"

    save_config(arch;
        rules = [("RGradCapped", () -> RGradCapped(γ1 = G1, γ2 = G2, γ3 = G3,
                                                   μ = 1.0, μ_max = 1.0)),
                 ("RDFO",  () -> RDFO(γ1 = G1, γ2 = G2, γ3 = G3, ζ = 1.0)),
                 ("RGrad", () -> RGrad(γ1 = G1, γ2 = G2, γ3 = G3, μ = 1.0))],
        models = [("ExactHessian", () -> ExactHessian())],
        subsolvers = [("ExactMS", () -> ExactMS()), ("SteihaugCG", () -> SteihaugCG())],
        params = sd_params(),
        extra = Dict("experiment" => "exp16_sinc_dichotomy",
                     "minimisers" => SINC_J, "offset" => SINC_OFFSET,
                     "kmax_below" => SD_KMAX_BELOW, "kmax_above" => SD_KMAX_ABOVE,
                     "tol" => SD_TOL,
                     "roots_source" => joinpath(sinc_data_dir(), "roots.dat")))

    RES = Dict{Symbol, Any}()

    # ---- 5.1 capped, the exact threshold ------------------------------------
    @info "5.1 RGradCapped, bisection on mu_bar"
    mu = NamedTuple[]
    for j in SINC_J
        λ = roots[j].λ
        mk(m) = RGradCapped(γ1 = G1, γ2 = G2, γ3 = G3, μ = min(1.0, m), μ_max = m)
        b = sd_bisect(mk, j, roots, 0.1 / λ, 20.0 / λ)
        grid, cls = sd_scan(mk, j, roots, 0.1 / λ, 20.0 / λ)
        mono = sd_monotone(cls)
        # kappa_bar_empirical defaults to the inactive iterations alone, which a
        # below-threshold run has none of, so it returns NaN there. We report the
        # above-threshold run, where the realised constant is what the analysis
        # bounds by 8/lambda*.
        ra = sd_run(mk(4.0 / λ), j, roots; kmax = SD_KMAX_ABOVE)
        rb = sd_run(mk(0.5 / λ), j, roots)
        push!(mu, (j = j, λ = λ, star = b.star, ok = b.ok, scaled = b.star * λ,
                   kbar = kappa_bar(λ), mono = mono, grid = grid, cls = cls,
                   kbar_emp = ra.kbar_emp, kbar_emp_all = rb.kbar_emp_all))
        @printf("  j=%d  mu*=%.6g  mu*lambda=%.4f  monotone=%s\n",
                j, b.star, b.star * λ, mono)
    end
    RES[:mu] = mu

    # ---- 5.2 RDFO, the located threshold ------------------------------------
    @info "5.2 RDFO, bisection on zeta and the Delta_0 sweep"
    ze = NamedTuple[]
    for j in SINC_J, Δ0 in (0.01, 0.1, 1.0)
        λ = roots[j].λ
        mk(z) = RDFO(γ1 = G1, γ2 = G2, γ3 = G3, ζ = z)
        vals, cls = sd_scan(mk, j, roots, 0.005 / λ, 100.0 / λ; n = 24, Δ0 = Δ0)
        # The coarse scan spans four decades and steps by a factor of 1.54, so it
        # cannot resolve a transition at zeta*lambda = 1/2. The fine scan is
        # linear in the scaled variable and does.
        fvals = collect(range(0.20, 2.00; length = 37)) ./ λ
        fcls  = [sd_class(sd_run(mk(v), j, roots; kmax = 400, Δ0 = Δ0))[1]
                 for v in fvals]
        fmono = sd_monotone(fcls)
        fi    = findfirst(c -> c !== :active, fcls)
        ftrans = fi === nothing ? NaN : fvals[fi] * λ
        fabove = fi === nothing ? :none : fcls[min(fi + 2, length(fcls))]
        mono = sd_monotone(cls)
        wlo, whi = sd_window(vals, cls)
        # A bisection is only run where the profile is monotone. Where it is not,
        # the transition is not a threshold and we report the window instead.
        b = mono ? sd_bisect(mk, j, roots, 0.005 / λ, 100.0 / λ; Δ0 = Δ0) :
                   (ok = false, star = NaN, lo = NaN, hi = NaN,
                    cls_lo = :na, cls_hi = :na, iters = 0)
        push!(ze, (j = j, λ = λ, Δ0 = Δ0, star = b.star, ok = b.ok,
                   scaled = b.star * λ, mono = mono, vals = vals, cls = cls,
                   wlo = wlo, whi = whi,
                   wlo_s = wlo === nothing ? NaN : wlo * λ,
                   whi_s = whi === nothing ? NaN : whi * λ,
                   n_neither = count(==(:neither), cls),
                   fvals = fvals, fcls = fcls, fmono = fmono,
                   ftrans = ftrans, fabove = fabove,
                   f_neither = count(==(:neither), fcls)))
        @printf("  j=%d D0=%.2f  transition zeta*l=%.4g  above=%-9s monotone=%-5s neither=%d\n",
                j, Δ0, ftrans, string(fabove), fmono, count(==(:neither), fcls))
    end
    RES[:zeta] = ze

    # ---- 5.3 uncapped, the boundedness of mu_k ------------------------------
    @info "5.3 RGrad uncapped, mu_infinity and the half-radius test"
    un = NamedTuple[]
    for j in SINC_J, μ0 in [0.01, 0.1, 1.0, 10.0] ./ roots[j].λ
        λ = roots[j].λ
        r = sd_run(RGrad(γ1 = G1, γ2 = G2, γ3 = G3, μ = μ0), j, roots;
                   kmax = SD_KMAX_ABOVE)
        μinf = isempty(r.theta) ? NaN : r.theta[end]
        push!(un, (j = j, λ = λ, μ0 = μ0, μinf = μinf, scaled = μinf * λ,
                   in_band = 2.0 <= μinf * λ < 2.0 * G3,
                   logmu0 = log(μ0 * λ) / log(G3), branches = r.branches,
                   iter = r.iter, status = r.status, limit_j = r.limit_j,
                   limit_dist = r.limit_dist, theta = r.theta,
                   step_err = r.step_err))
        @printf("  j=%d mu0=%.4g -> mu_inf=%.4g  scaled=%.4f  band=%s  limit j=%d\n",
                j, μ0, μinf, μinf * λ, 2.0 <= μinf * λ < 2.0 * G3, r.limit_j)
    end
    RES[:uncapped] = un

    # ---- 5.4 the second threshold, a measurement only -----------------------
    @info "5.4 the second threshold, measured and not built upon"
    sec = NamedTuple[]
    for j in SINC_J
        λ = roots[j].λ; found = NaN
        for m in exp10.(range(log10(0.5 / λ), log10(200.0 / λ); length = 40))
            r = sd_run(RGradCapped(γ1 = G1, γ2 = G2, γ3 = G3, μ = min(1.0, m),
                                   μ_max = m), j, roots; kmax = SD_KMAX_ABOVE)
            isempty(r.active) || (r.active[end] && (found = m))
        end
        push!(sec, (j = j, λ = λ, largest = found, predicted = 2.0 / λ,
                    ratio = found / (2.0 / λ)))
        @printf("  j=%d  largest binding mu_bar = %.4g   2/lambda = %.4g\n",
                j, found, 2.0 / λ)
    end
    RES[:second] = sec

    # ---- 6 precision --------------------------------------------------------
    @info "6 precision, the below-threshold arm in both precisions"
    setprecision(BigFloat, 192)
    pr = NamedTuple[]
    for j in SINC_J
        λ = roots[j].λ; m = 0.5 / λ
        for big in (false, true)
            r = sd_run(RGradCapped(γ1 = G1, γ2 = G2, γ3 = G3, μ = min(1.0, m),
                                   μ_max = m), j, roots; big = big)
            push!(pr, (j = j, λ = λ, mubar = m, big = big, iter = r.iter,
                       status = r.status, gnorm = r.gnorm, grad = r.grad,
                       cls = sd_class(r)[1], rate = abs(1 - m * λ),
                       limit_j = r.limit_j, limit_dist = r.limit_dist,
                       step_err = r.step_err))
            @printf("  j=%d %-9s iter=%5d %-12s |g|=%.3e predicted rate %.4f\n",
                    j, big ? "BigFloat" : "Float64", r.iter, string(r.status),
                    r.gnorm, abs(1 - m * λ))
        end
    end
    RES[:precision] = pr

    # ---- traces for the dichotomy figure ------------------------------------
    tr = NamedTuple[]
    for j in SINC_J
        λ = roots[j].λ
        for (side, m) in ((:below, 0.5 / λ), (:above, 4.0 / λ))
            r = sd_run(RGradCapped(γ1 = G1, γ2 = G2, γ3 = G3, μ = min(1.0, m),
                                   μ_max = m), j, roots;
                       kmax = side === :below ? 400 : SD_KMAX_ABOVE)
            push!(tr, (rule = "RGradCapped", j = j, side = side, param = m,
                       theta_a = r.theta_a, cls = sd_class(r)[1],
                       step_err = r.step_err, n_negcurv = r.n_negcurv))
        end
        for (side, z) in ((:below, 0.1 / λ), (:above, 2.0 / λ))
            r = sd_run(RDFO(γ1 = G1, γ2 = G2, γ3 = G3, ζ = z), j, roots;
                       kmax = side === :below ? 400 : SD_KMAX_ABOVE)
            push!(tr, (rule = "RDFO", j = j, side = side, param = z,
                       theta_a = r.theta_a, cls = sd_class(r)[1],
                       step_err = r.step_err, n_negcurv = r.n_negcurv))
        end
    end
    RES[:traces] = tr

    sd_tables(arch, RES, roots)
    sd_figures(arch, RES, roots)
    save_data(arch, "sinc_dichotomy.jld2";
              mu = RES[:mu], zeta = RES[:zeta], uncapped = RES[:uncapped],
              second = RES[:second])
    finalize_archive(arch; notes = """
        The activity dichotomy on sin(x)/x, minimisers j = $(SINC_J).

        This locates the true constant inside the interval on which neither of
        Part II's theorems speaks. It does not confirm a proved dichotomy.

        Critical points read from task 1 at
        $(joinpath(sinc_data_dir(), "roots.dat")). Settings from
        benchmark/config.jl, the settings subsection of the paper carrying a TODO
        rather than a table.
        """)
    return arch, RES
end

# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------
function sd_tables(arch, RES, roots)
    io = IOBuffer()
    println(io, "Table 1. Per minimiser. Every constant of the first four columns")
    println(io, "comes from task 1 and is not recomputed here.")
    println(io)
    @printf(io, "%3s %14s %12s %12s %12s %12s %10s %12s %10s %12s\n", "j", "y_j",
            "lambda*", "1/lambda*", "kbar=8/l", "mu*", "mu*.l", "zeta*", "zeta*.l",
            "kbar_emp")
    println(io, "-"^122)
    for j in SINC_J
        m = only(r for r in RES[:mu] if r.j == j)
        z = only(r for r in RES[:zeta] if r.j == j && r.Δ0 == 0.01)
        @printf(io, "%3d %14.9f %12.8f %12.6f %12.4f %12.6g %10.4f %12.6g %10.4f %12.4g\n",
                j, roots[j].y, roots[j].λ, 1 / roots[j].λ, m.kbar,
                m.star, m.scaled, z.ftrans / roots[j].λ, z.ftrans, m.kbar_emp)
    end
    println(io)
    println(io, "mu*.l is predicted to be 1. The RDFO column is NOT a threshold:")
    println(io, "the classification is not monotone in zeta, so the column reports the")
    println(io, "lower edge of the window on which the run classifies as eventually")
    println(io, "inactive, at Delta_0 = 0.01. See exp16_tab6_rdfo_scan.txt.")
    println(io, "kbar = 8/lambda* is the constant the analysis carries. kbar_emp is the")
    println(io, "largest realised |s_k|/|g_k| on an above-threshold run, and the ratio")
    println(io, "of the two is the factor the analysis has to spare.")
    save_table(arch, "exp16_tab1_per_minimiser.txt", String(take!(io)))

    io = IOBuffer()
    println(io, "Table 2. Per configuration. Three classes, never two.")
    println(io, "Iteration counts of the below and above arms are not comparable")
    println(io, "and carry different budgets, stated in the last column.")
    println(io)
    @printf(io, "%-12s %3s %6s %10s %9s %12s %8s %8s %8s %12s %6s %10s\n",
            "rule", "j", "side", "param", "prec", "class", "inact", "accept",
            "reject", "stop", "lim j", "dist")
    println(io, "-"^124)
    for t in RES[:traces]
        @printf(io, "%-12s %3d %6s %10.4g %9s %12s %8s %8s %8s %12s %6s %10s\n",
                t.rule, t.j, string(t.side), t.param, "Float64", string(t.cls),
                "", "", "", "", "", "")
    end
    for p in RES[:precision]
        @printf(io, "%-12s %3d %6s %10.4g %9s %12s %8s %8s %8s %12s %6d %10.2e\n",
                "RGradCapped", p.j, "below", p.mubar, p.big ? "BigFloat" : "Float64",
                string(p.cls), "n/a", "", "", string(p.status), p.limit_j,
                p.limit_dist)
    end
    for u in RES[:uncapped]
        @printf(io, "%-12s %3d %6s %10.4g %9s %12s %8s %8s %8s %12s %6d %10.2e\n",
                "RGrad", u.j, "n/a", u.μ0, "Float64", "n/a", "", "", "",
                string(u.status), u.limit_j, u.limit_dist)
    end
    save_table(arch, "exp16_tab2_per_configuration.txt", String(take!(io)))

    io = IOBuffer()
    println(io, "5.2 RDFO. The classification across zeta, at three values of Delta_0.")
    println(io, "The profile is not monotone in zeta, so there is no threshold to")
    println(io, "bisect for. `a` is eventually active, `i` eventually inactive and")
    println(io, "`n` neither, which is the third class the design requires.")
    println(io)
    for z in RES[:zeta]
        @printf(io, "j=%d  Delta_0=%-6g  transition at zeta*lambda = %.4g,  class just above = %s\n", z.j, z.Δ0, z.ftrans, string(z.fabove))
        @printf(io, "     monotone over [0.2, 2.0]: %-6s   neither cells: %d\n",
                z.fmono, z.f_neither)
        @printf(io, "     zeta*l: ")
        for v in z.fvals; @printf(io, "%5.2f", v * z.λ); end
        println(io)
        @printf(io, "     class : ")
        for c in z.fcls
            @printf(io, "%5s", c === :active ? "a" : c === :inactive ? "i" : "n")
        end
        println(io); println(io)
    end
    println(io, "Delta_0 is decisive for RDFO and irrelevant to RGradCapped by")
    println(io, "construction, which is the point of this sweep. Where a window exists")
    println(io, "its lower edge sits near zeta*lambda = 1/2.")
    save_table(arch, "exp16_tab6_rdfo_scan.txt", String(take!(io)))

    io = IOBuffer()
    println(io, "5.3 The boundedness of the multiplier under uncapped RGrad.")
    println(io, "Prediction: mu_inf * lambda* lies in [2, 2*gamma3) = [2, ", 2G3, ").")
    println(io)
    @printf(io, "%3s %12s %12s %10s %8s %14s %8s\n", "j", "mu_0", "mu_inf",
            "scaled", "in band", "log_g3(mu0.l)", "limit j")
    println(io, "-"^82)
    nin = 0
    for u in RES[:uncapped]
        u.in_band && (nin += 1)
        @printf(io, "%3d %12.5g %12.5g %10.4f %8s %14.3f %8d\n",
                u.j, u.μ0, u.μinf, u.scaled, u.in_band, u.logmu0, u.limit_j)
    end
    @printf(io, "\n  %d of %d configurations land in the band.\n", nin,
            length(RES[:uncapped]))
    println(io, "\nBranch counts, the direct evidence for the half-radius test:")
    for u in RES[:uncapped]
        @printf(io, "  j=%d mu0=%.4g  %s\n", u.j, u.μ0, string(u.branches))
    end
    save_table(arch, "exp16_tab3_uncapped.txt", String(take!(io)))

    io = IOBuffer()
    println(io, "5.4 The second threshold. A measurement, not a result.")
    println(io, "The largest mu_bar at which the cap still binds at the last")
    println(io, "iteration, against 2/lambda*.")
    println(io)
    @printf(io, "%3s %14s %14s %10s\n", "j", "largest binding", "2/lambda*", "ratio")
    println(io, "-"^46)
    for s in RES[:second]
        @printf(io, "%3d %14.6g %14.6g %10.4f\n", s.j, s.largest, s.predicted, s.ratio)
    end
    save_table(arch, "exp16_tab4_second_threshold.txt", String(take!(io)))

    io = IOBuffer()
    println(io, "6. Precision. The below-threshold arm at mu_bar = 0.5/lambda*.")
    println(io, "The theorem gives linear convergence at rate |1 - mu_bar*lambda*|")
    println(io, "with the constraint active at every iteration. What stops the")
    println(io, "Float64 run is the achieved reduction falling to the rounding")
    println(io, "level of f, which is an arithmetic event and not that theorem.")
    println(io)
    @printf(io, "%3s %10s %8s %8s %14s %12s %10s\n", "j", "precision", "iter",
            "class", "status", "final |g|", "rate")
    println(io, "-"^76)
    for p in RES[:precision]
        @printf(io, "%3d %10s %8d %8s %14s %12.4e %10.4f\n", p.j,
                p.big ? "BigFloat" : "Float64", p.iter, string(p.cls),
                string(p.status), p.gnorm, p.rate)
    end
    save_table(arch, "exp16_tab5_precision.txt", String(take!(io)))

    io = IOBuffer()
    println(io, "Hazard checks.")
    println(io)
    println(io, "1. The closed-form one-dimensional step. The largest relative")
    println(io, "   discrepancy between the returned |s_k| and -g/a inside the")
    println(io, "   region, or Delta on the boundary, over every traced run:")
    w = maximum(t.step_err for t in RES[:traces])
    wn = sum(t.n_negcurv for t in RES[:traces])
    @printf(io, "     %.3e over %d traced runs, %d iterations with a_k <= 0\n",
            w, length(RES[:traces]), wn)
    println(io, w < 1e-8 ? "     AGREES" : "     DOES NOT AGREE")
    println(io)
    println(io, "2. The limit point of every run, as the nearest critical point.")
    println(io, "   Reported per configuration in Table 2 and Table 3.")
    println(io)
    println(io, "3. Iteration budgets. Below-threshold runs carry ",
            SD_KMAX_BELOW, " iterations,")
    println(io, "   above-threshold runs ", SD_KMAX_ABOVE,
            ". Their counts never share a column.")
    save_table(arch, "exp16_hazards.txt", String(take!(io)))
    return nothing
end

# -----------------------------------------------------------------------------
# Figures. Black plus one accent, separated by dash pattern too.
# -----------------------------------------------------------------------------
const SD_BLACK  = RGB(0.0, 0.0, 0.0)
const SD_ACCENT = RGB(0.85, 0.33, 0.10)
const SD_GREY   = RGB(0.60, 0.60, 0.60)

function sd_figures(arch, RES, roots)
    dash = (:solid, :dash, :dot, :dashdot)

    # 1. the dichotomy: theta_k a_k against k, one panel per rule
    panels = []
    for rule in ("RGradCapped", "RDFO")
        p = plot(xlabel = "k", ylabel = "θ_k a_k", yscale = :log10,
                 title = rule, legend = :right, size = (620, 430))
        for (i, j) in enumerate(SINC_J), (side, ls) in ((:below, :solid), (:above, :dash))
            t = [q for q in RES[:traces] if q.rule == rule && q.j == j && q.side == side]
            isempty(t) && continue
            v = filter(x -> isfinite(x) && x > 0, t[1].theta_a)
            isempty(v) && continue
            plot!(p, 1:length(v), v; label = "j=$j $(side)",
                  color = isodd(i) ? SD_BLACK : SD_ACCENT,
                  linestyle = ls, lw = 1.4)
        end
        hline!(p, [1.0]; color = SD_GREY, linestyle = :dot, lw = 2, label = "θa = 1")
        push!(panels, p)
    end
    savefig_archived(arch, "exp16_fig1_dichotomy.pdf",
                     plot(panels...; layout = (1, 2), size = (1240, 430)))

    # 2. the threshold is a constant of the solution
    let inv = [1 / roots[j].λ for j in SINC_J]
        p = plot(xlabel = "1/λ*_j", ylabel = "threshold", legend = :topleft,
                 size = (640, 440))
        ms = [only(r for r in RES[:mu] if r.j == j).star for j in SINC_J]
        zs = [let q = only(r for r in RES[:zeta] if r.j == j && r.Δ0 == 0.01)
                  q.ftrans / roots[j].λ
              end for j in SINC_J]
        plot!(p, inv, ms; label = "μ̄* (R-grad capped)", color = SD_BLACK,
              marker = :circle, ms = 5, lw = 1.6)
        plot!(p, inv, zs; label = "ζ* (R-DFO transition)", color = SD_ACCENT,
              marker = :square, ms = 5, lw = 1.6, linestyle = :dash)
        plot!(p, inv, inv; label = "level 1", color = SD_GREY, linestyle = :dot, lw = 2)
        plot!(p, inv, inv ./ 2; label = "level 1/2", color = SD_GREY,
              linestyle = :dashdot, lw = 1.5)
        plot!(p, inv, inv ./ 5; label = "level 1/5", color = SD_GREY,
              linestyle = :dash, lw = 1.5)
        savefig_archived(arch, "exp16_fig2_thresholds.pdf", p)
    end

    # 3. the sawtooth
    let p = plot(xlabel = "log_{γ3}(μ_0 λ*_j)", ylabel = "μ_∞ λ*_j",
                 legend = :topright, size = (640, 440))
        for (i, j) in enumerate(SINC_J)
            u = [q for q in RES[:uncapped] if q.j == j]
            isempty(u) && continue
            plot!(p, [q.logmu0 for q in u], [q.scaled for q in u]; label = "j=$j",
                  color = isodd(i) ? SD_BLACK : SD_ACCENT,
                  linestyle = dash[mod1(i, 4)], marker = :circle, ms = 4, lw = 1.5)
        end
        hline!(p, [2.0]; color = SD_GREY, linestyle = :dot, lw = 2, label = "2")
        hline!(p, [2G3]; color = SD_GREY, linestyle = :dash, lw = 2, label = "2γ3")
        savefig_archived(arch, "exp16_fig3_sawtooth.pdf", p)
    end

    # 4. the branch trace of one uncapped run, as theta_k against k
    let u = RES[:uncapped][1]
        p = plot(xlabel = "k", ylabel = "θ_k = μ_k", legend = :bottomright,
                 size = (640, 420), title = "uncapped R-grad, j=$(u.j), μ_0=$(round(u.μ0, sigdigits=3))")
        plot!(p, 0:length(u.theta)-1, u.theta; label = "μ_k", color = SD_BLACK, lw = 1.6)
        hline!(p, [2 / u.λ]; color = SD_ACCENT, linestyle = :dash, lw = 2,
               label = "2/λ*, where the half-radius test fails")
        savefig_archived(arch, "exp16_fig4_branches.pdf", p)
    end

    # 5. precision
    let p = plot(xlabel = "k", ylabel = "‖g_k‖", yscale = :log10,
                 legend = :bottomleft, size = (640, 440))
        for (i, j) in enumerate(SINC_J[1:min(2, end)])
            for q in RES[:precision]
                q.j == j || continue
                v = filter(x -> isfinite(x) && x > 0, q.grad)
                isempty(v) && continue
                plot!(p, 1:length(v), v;
                      label = "j=$j $(q.big ? "BigFloat" : "Float64")",
                      color = q.big ? SD_ACCENT : SD_BLACK,
                      linestyle = q.big ? :dash : :solid, lw = 1.5)
            end
        end
        savefig_archived(arch, "exp16_fig5_precision.pdf", p)
    end
    return nothing
end
