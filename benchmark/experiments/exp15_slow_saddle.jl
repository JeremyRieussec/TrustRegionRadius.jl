# =============================================================================
# benchmark/experiments/exp15_slow_saddle.jl
#
# EXPERIMENT 15 -- a criticality-anchored radius cannot escape a slow saddle in
# bounded time, with perfect curvature information.
#
# On the family
#
#     f_eps(x,y) = ½x² + ε(¼y⁴ − ½y²),
#
# the origin is a saddle with ∇²f = diag(1, −ε), nonsingular for every ε > 0,
# and the two minimisers sit at (0, ±1), at distance 1 from the saddle whatever
# ε is. So ε tunes the escape rate without moving the geometry.
#
# WHAT IS CLAIMED, AND WHAT THE CLAIM RESTS ON
#
# The line {x = 0} is invariant: the gradient lies along e2 and the Hessian is
# diagonal, so every CG iterate stays along e2. On that line, for |y| ≤ 1,
#
#     ‖g‖ = ε|y|(1 − y²) ≤ ε|y| .
#
# Every trust-region step satisfies ‖s_k‖ ≤ Δ_k. If in addition the radius rule
# satisfies, for some C > 0 and k0,
#
#     Δ_k ≤ C · max_{k0 ≤ j ≤ k} ‖g_j‖                              (R.CA)
#
# then, writing M_k = max_{j ≤ k}|y_j|, we get M_{k+1} ≤ M_k(1 + Cε) up to the
# escape threshold, hence
#
#     k_esc  >  ln(1/(2|y_0|)) / ln(1 + Cε) .
#
# RGradCapped satisfies (R.CA) with C = μ̄ and k0 = 0. RDFO satisfies it with
# C = γ3ζ from the first index at which Δ ≤ ζ‖g‖, and that entry condition is
# not automatic: a run that never enters the band contracts the radius
# geometrically and converges to the saddle instead.
#
# The bound uses NO property of the model or of the subproblem solution beyond
# ‖s_k‖ ≤ Δ_k. That is what makes it a statement about the radius axis. The
# model here is the true Hessian throughout, so the stall appears with perfect
# curvature information.
#
# THE PREDICTION THAT SEPARATES THE TWO MEASURES
#
# With ω_k = τ_k = max{‖g_k‖, −λ⁻_k}, near the saddle −λ⁻ = ε while
# ‖g_k‖ ≈ ε|y_k|, so τ_k ≈ ε and the radius is about μ̄ε, a constant rather than
# a quantity proportional to |y_k|. Growth becomes arithmetic and
# k_esc ≈ 1/(2μ̄ε). The ratio of the two escape times should then be about
# 2ln(1/(2y_0)), independent of ε. That ratio is fitted here, not assumed.
#
# TWO TOLERANCE REGIMES, REPORTED SEPARATELY
#
# Near the saddle ‖g_k‖ ≈ ε|y_k|, so for ε|y_0| below the first-order tolerance
# the method stops at the saddle and reports success. The grid is therefore run
# twice. The RATE regime uses tol = 1e-14, which lets the dynamics run so that
# k_esc and the geometric rate are measurable. The OUTCOME regime uses
# tol = 1e-5, the value SOLVER_PARAMS carries, which is where false convergence
# appears. Neither tolerance was chosen to make a prediction hold, and both are
# reported.
#
#   julia --project=benchmark -e 'include("benchmark/initialisation.jl"); slow_saddle()'
#
# No `using` and no `include` here: initialisation.jl loads the packages,
# harness.jl, archive.jl and config.jl once, in order.
# =============================================================================

# -----------------------------------------------------------------------------
# The grid
# -----------------------------------------------------------------------------
const SS_EPS   = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
const SS_MUBAR = [0.01, 0.1, 1.0, 10.0]
const SS_ZETA  = [0.01, 0.1, 1.0, 10.0]
const SS_Y0    = [1e-3, 1e-6]
const SS_X0    = [0.0, 0.8]

"The escape threshold of the proposition. |y_k| > ½ is escape."
const SS_THRESH = 0.5

"Iterations allowed. A run that has not escaped by then is reported `never`."
const SS_KMAX = 50_000

"Below this k_esc we call the escape fast. Stated so the outcome map is readable."
const SS_FAST = 10

const SS_TOL_RATE    = 1e-14
const SS_TOL_OUTCOME = 1e-5

ss_params(tol; tol_H = -1.0, Δ0 = 1.0) =
    TRParams(η = 0.1, η1 = 0.1, η2 = 0.9, Δ0 = Δ0, Δmin = 0.0, Δmax = Inf,
             tol = tol, tol_H = tol_H, max_iterations = SS_KMAX, max_time = 300.0)

# Factories. Keyword arguments everywhere, since RGrad and RGradCapped were
# renumbered and a positional call can pass under the wrong reading.
ss_rules(μ̄, ζ) = [
    ("RDelta",         :comparator, NaN, () -> RDelta(γ1 = G1, γ2 = G2, γ3 = G3)),
    ("RGradCapped",    :grad,       μ̄,   () -> RGradCapped(γ1 = G1, γ2 = G2, γ3 = G3,
                                                           μ = min(1.0, μ̄), μ_max = μ̄)),
    ("RDFO",           :dfo,        ζ,   () -> RDFO(γ1 = G1, γ2 = G2, γ3 = G3, ζ = ζ)),
    ("RGradCappedTau", :gradtau,    μ̄,   () -> RGradCappedTau(γ1 = G1, γ2 = G2, γ3 = G3,
                                                              μ = min(1.0, μ̄), μ_max = μ̄)),
    ("RDFOTau",        :dfotau,     ζ,   () -> RDFOTau(γ1 = G1, γ2 = G2, γ3 = G3, ζ = ζ)),
]

# -----------------------------------------------------------------------------
# One run
# -----------------------------------------------------------------------------

"""
    ss_run(rule, ε, y0, x0, params) -> NamedTuple

One `tr_solve` call on `flat_well(ε)`, reduced to what the proposition needs.

The iterate path comes from the solver's own callback, which is the only way to
reach `y_k`. The model is opened here rather than through `run_experiment` so
that the evaluation counters survive, since the extra curvature cost of `τ_k` is
one of the quantities asked for.
"""
function ss_run(rule, ε, y0, x0, params)
    nlp = flat_well(ε; x0 = [x0, y0])
    xs = [Float64[x0, y0]]
    t0 = time()
    st = tr_solve(nlp; rule = rule, model = ExactHessian(), subsolver = SteihaugCG(),
                  params = params, trace = true,
                  callback = (_n, _s, s) -> push!(xs, copy(s.solution)))
    dt = time() - t0
    ss = st.solver_specific
    Δ  = get(ss, :delta_trajectory, Float64[])
    g  = get(ss, :grad_trajectory,  Float64[])
    ac = get(ss, :active_trajectory, Bool[])

    ys = [p[2] for p in xs]
    xsx = [abs(p[1]) for p in xs]
    kesc = findfirst(v -> abs(v) > SS_THRESH, ys)          # 1-based on xs
    kesc = kesc === nothing ? nothing : kesc - 1           # iteration index

    # The escape phase: the x component has been solved and the iterate has not
    # yet escaped. On the x0 = 0 arm |x_k| is identically zero and the phase is
    # the whole pre-escape run. The rule is stated rather than tuned.
    hi = kesc === nothing ? length(ys) : kesc + 1
    phase = [i for i in 1:hi if xsx[i] <= ε * max(abs(ys[i]), 1e-300) &&
                                abs(ys[i]) <= SS_THRESH && ys[i] != 0]
    rate = NaN; rate_n = 0
    if length(phase) >= 5
        K = Float64.(phase .- 1); Y = log.(abs.(ys[phase]))
        K̄, Ȳ = sum(K)/length(K), sum(Y)/length(Y)
        sxx = sum(abs2, K .- K̄)
        sxx > 0 && (rate = sum((K .- K̄) .* (Y .- Ȳ)) / sxx)
        rate_n = length(phase)
    end

    θ = (isempty(Δ) || isempty(g)) ? Float64[] :
        [g[i] > 0 ? Δ[i]/g[i] : NaN for i in eachindex(g)]
    dsad = norm([xs[end][1], xs[end][2]])
    dmin = minimum(norm(xs[end] .- [0.0, s]) for s in (1.0, -1.0))

    return (kesc = kesc, rate = rate, rate_n = rate_n,
            status = st.status, iter = st.iter, time = dt,
            f_evals = neval_obj(nlp), g_evals = neval_grad(nlp),
            h_evals = neval_hprod(nlp),
            y_end = ys[end], x_end = xs[end][1],
            gnorm = Float64(st.dual_feas),
            dist_saddle = dsad, dist_min = dmin,
            theta_max = isempty(θ) ? NaN : maximum(x -> isfinite(x) ? x : -Inf, θ),
            theta_med = isempty(θ) ? NaN : sort(filter(isfinite, θ))[max(1, end ÷ 2)],
            active = isempty(ac) ? NaN : count(ac)/length(ac),
            n_small_g = count(<(1e-5), g),
            delta_0 = isempty(Δ) ? NaN : Δ[1],
            ys = ys, deltas = Δ)
end

"""
    ss_class(r, tol) -> Symbol

The three outcomes of the design, by a stated rule.

`:false_convergence` is a run that met its first-order criterion without leaving
the neighbourhood of the saddle. `:fast` and `:slow` split the escapes at
`SS_FAST`. `:never` is a run that neither escaped nor converged.
"""
function ss_class(r)
    if r.kesc === nothing
        return (r.status === :first_order && abs(r.y_end) <= SS_THRESH) ?
               :false_convergence : :never
    end
    return r.kesc <= SS_FAST ? :fast : :slow
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function slow_saddle()
    arch = ExperimentArchive(tag = "slow_saddle")
    save_config(arch; rules = vcat([(nm, f) for (nm, _, _, f) in ss_rules(1.0, 1.0)]),
                models = [("ExactHessian", () -> ExactHessian())],
                subsolvers = [("SteihaugCG", () -> SteihaugCG())],
                params = ss_params(SS_TOL_RATE),
                extra = Dict("experiment" => "exp15_slow_saddle",
                             "eps" => SS_EPS, "mu_bar" => SS_MUBAR,
                             "zeta" => SS_ZETA, "y0" => SS_Y0, "x0" => SS_X0,
                             "threshold" => SS_THRESH, "kmax" => SS_KMAX,
                             "tol_rate" => SS_TOL_RATE,
                             "tol_outcome" => SS_TOL_OUTCOME,
                             "fast_cut" => SS_FAST))

    rows = NamedTuple[]
    for (regime, tol) in ((:rate, SS_TOL_RATE), (:outcome, SS_TOL_OUTCOME))
        p = ss_params(tol)
        for ε in SS_EPS, y0 in SS_Y0, x0 in SS_X0
            for (nm, kind, param, mk) in ss_rules(1.0, 1.0)
                kind === :comparator || continue
                r = ss_run(mk(), ε, y0, x0, p)
                push!(rows, merge((regime = regime, rule = nm, kind = kind,
                                   param = NaN, ε = ε, y0 = y0, x0 = x0,
                                   tol = tol, tol_H = -1.0), r))
            end
            for μ̄ in SS_MUBAR, (nm, kind, param, mk) in ss_rules(μ̄, 1.0)
                kind in (:grad, :gradtau) || continue
                r = ss_run(mk(), ε, y0, x0, p)
                push!(rows, merge((regime = regime, rule = nm, kind = kind,
                                   param = μ̄, ε = ε, y0 = y0, x0 = x0,
                                   tol = tol, tol_H = -1.0), r))
            end
            for ζ in SS_ZETA, (nm, kind, param, mk) in ss_rules(1.0, ζ)
                kind in (:dfo, :dfotau) || continue
                r = ss_run(mk(), ε, y0, x0, p)
                push!(rows, merge((regime = regime, rule = nm, kind = kind,
                                   param = ζ, ε = ε, y0 = y0, x0 = x0,
                                   tol = tol, tol_H = -1.0), r))
            end
        end
        @info "regime done" regime nrows = length(rows)
    end

    # --- RDFO started inside the criticality-controlled band -----------------
    # The proposition's RDFO branch needs an index k0 with Delta_{k0} <= zeta*||g_{k0}||.
    # From Delta_0 = 1 that never happens before escape: the first step is
    # enormous relative to ||g_0|| ~ eps*y0 and the run leaves the neighbourhood
    # at k = 1. The entry condition is therefore not a formality, and testing the
    # branch at all requires starting inside the band. This arm sets
    # Delta_0 = zeta*||g_0||, which satisfies entry at k0 = 0 by construction.
    rows_entry = NamedTuple[]
    for ε in SS_EPS, y0 in SS_Y0, ζ in SS_ZETA
        g0 = ε * abs(y0) * (1 - y0^2)
        Δ0 = ζ * g0
        Δ0 > 0 || continue
        r = ss_run(RDFO(γ1 = G1, γ2 = G2, γ3 = G3, ζ = ζ), ε, y0, 0.0,
                   ss_params(SS_TOL_RATE; Δ0 = Δ0))
        push!(rows_entry, merge((regime = :dfo_entry, rule = "RDFO", kind = :dfo,
                                 param = ζ, ε = ε, y0 = y0, x0 = 0.0,
                                 tol = SS_TOL_RATE, tol_H = -1.0), r))
    end
    append!(rows, rows_entry)
    @info "RDFO entry arm done" ncells = length(rows_entry)

    # --- the false-convergence cells, rerun with the second-order test on -----
    # A method that refuses to stop at a saddle should escape slowly instead,
    # which turns a wrong answer into an expensive one. That is the claim tested
    # here. tol_H is the stopping test and is a different use of second-order
    # information from the radius measure, so the two are never pooled below.
    fc = [r for r in rows if r.regime === :outcome && ss_class(r) === :false_convergence]
    rows2 = NamedTuple[]
    for r in fc
        mk = only(f for (nm, _, _, f) in ss_rules(r.param, r.param) if nm == r.rule)
        rr = ss_run(mk(), r.ε, r.y0, r.x0, ss_params(SS_TOL_OUTCOME; tol_H = 1e-6))
        push!(rows2, merge((regime = :second_order, rule = r.rule, kind = r.kind,
                            param = r.param, ε = r.ε, y0 = r.y0, x0 = r.x0,
                            tol = SS_TOL_OUTCOME, tol_H = 1e-6), rr))
    end
    append!(rows, rows2)
    @info "second-order rerun done" ncells = length(rows2)

    save_data(arch, "slow_saddle_rows.jld2";
              rows = [Base.structdiff(r, (ys = 0, deltas = 0)) for r in rows])
    ss_tables(arch, rows)
    ss_figures(arch, rows)
    finalize_archive(arch; notes = """
        Slow saddles on the radius axis. f(x,y) = ½x² + ε(¼y⁴ − ½y²), true
        Hessian and truncated CG throughout, so the stall appears with perfect
        curvature information.

        Two tolerance regimes, never pooled. `rate` at tol = $(SS_TOL_RATE) lets
        the dynamics run so that k_esc and the geometric rate are measurable.
        `outcome` at tol = $(SS_TOL_OUTCOME) is where false convergence appears.
        `second_order` reruns the false-convergence cells with tol_H = 1e-6.

        The radius measure ω_k and the stopping test tol_H are two different uses
        of second-order information and are reported separately throughout.
        """)
    return arch, rows
end

# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------
function ss_tables(arch, rows)
    sel(; kw...) = [r for r in rows if all(getfield(r, k) == v for (k, v) in kw)]
    kesc_s(r) = r.kesc === nothing ? "never" : string(r.kesc)

    io = IOBuffer()
    println(io, "The (mu_bar, eps) grid on the invariant line x0 = 0, rate regime.")
    println(io, "k_esc is the first iteration with |y_k| > $(SS_THRESH).")
    println(io, "`rate` is the fitted slope of log|y_k| on k over the escape phase,")
    println(io, "to be compared with ln(1 + mu_bar*eps) in the next column.")
    println(io)
    @printf(io, "%6s %8s %8s %9s %11s %11s %11s %9s %8s\n", "y0", "mu_bar",
            "eps", "k_esc", "rate", "ln(1+mu*e)", "mu*e*k_esc", "theta", "active")
    println(io, "-"^92)
    for y0 in SS_Y0, μ̄ in SS_MUBAR, ε in SS_EPS
        rs = sel(regime = :rate, rule = "RGradCapped", param = μ̄, ε = ε, y0 = y0, x0 = 0.0)
        isempty(rs) && continue
        r = rs[1]
        @printf(io, "%6.0e %8g %8.0e %9s %11.5f %11.5f %11s %9.4f %8.3f\n",
                y0, μ̄, ε, kesc_s(r), r.rate, log1p(μ̄*ε),
                r.kesc === nothing ? "n/a" : @sprintf("%.4f", μ̄*ε*r.kesc),
                r.theta_med, r.active)
    end
    save_table(arch, "exp15_grid_grad.txt", String(take!(io)))

    io = IOBuffer()
    println(io, "The (zeta, eps) grid, RDFO, rate regime, x0 = 0.")
    println(io, "The predicted constant is C = gamma3*zeta = $(G3)*zeta.")
    println(io)
    @printf(io, "%6s %8s %8s %9s %11s %11s %11s %9s\n", "y0", "zeta", "eps",
            "k_esc", "rate", "ln(1+g3*z*e)", "g3*z*e*kesc", "theta")
    println(io, "-"^84)
    for y0 in SS_Y0, ζ in SS_ZETA, ε in SS_EPS
        rs = sel(regime = :rate, rule = "RDFO", param = ζ, ε = ε, y0 = y0, x0 = 0.0)
        isempty(rs) && continue
        r = rs[1]; C = G3*ζ
        @printf(io, "%6.0e %8g %8.0e %9s %11.5f %11.5f %11s %9.4f\n",
                y0, ζ, ε, kesc_s(r), r.rate, log1p(C*ε),
                r.kesc === nothing ? "n/a" : @sprintf("%.4f", C*ε*r.kesc), r.theta_med)
    end
    save_table(arch, "exp15_grid_dfo.txt", String(take!(io)))

    io = IOBuffer()
    println(io, "RDFO started inside the criticality-controlled band, Delta_0 = zeta*||g_0||,")
    println(io, "so that the entry condition of the proposition holds at k0 = 0.")
    println(io, "The predicted constant is C = gamma3*zeta = $(G3)*zeta.")
    println(io, "Compare with exp15_grid_dfo.txt, where Delta_0 = 1 puts the run")
    println(io, "outside the band and it escapes at k = 1 whatever zeta and eps are.")
    println(io)
    @printf(io, "%6s %8s %8s %9s %11s %13s %13s %9s\n", "y0", "zeta", "eps",
            "k_esc", "rate", "ln(1+g3*z*e)", "g3*z*e*kesc", "theta")
    println(io, "-"^86)
    for y0 in SS_Y0, ζ in SS_ZETA, ε in SS_EPS
        rs = sel(regime = :dfo_entry, param = ζ, ε = ε, y0 = y0)
        isempty(rs) && continue
        r = rs[1]; C = G3*ζ
        @printf(io, "%6.0e %8g %8.0e %9s %11.5f %13.5f %13s %9.4f\n",
                y0, ζ, ε, kesc_s(r), r.rate, log1p(C*ε),
                r.kesc === nothing ? "n/a" : @sprintf("%.4f", C*ε*r.kesc), r.theta_med)
    end
    save_table(arch, "exp15_grid_dfo_entry.txt", String(take!(io)))

    io = IOBuffer()
    println(io, "The comparator. RDelta from the same starting points.")
    println(io, "Escape should take a bounded number of iterations, independent of eps.")
    println(io)
    @printf(io, "%6s %6s %8s %9s %11s %9s\n", "y0", "x0", "eps", "k_esc", "status", "iter")
    println(io, "-"^56)
    for y0 in SS_Y0, x0 in SS_X0, ε in SS_EPS
        rs = sel(regime = :rate, rule = "RDelta", ε = ε, y0 = y0, x0 = x0)
        isempty(rs) && continue
        r = rs[1]
        @printf(io, "%6.0e %6.1f %8.0e %9s %11s %9d\n", y0, x0, ε, kesc_s(r),
                string(r.status), r.iter)
    end
    save_table(arch, "exp15_comparator.txt", String(take!(io)))

    io = IOBuffer()
    println(io, "The two criticality measures. ||g|| against tau, same mu_bar.")
    println(io, "The prediction is geometric growth against arithmetic growth, so")
    println(io, "the ratio of escape times should be about 2 ln(1/(2 y0)),")
    println(io, "independent of eps. The fitted ratio is in the last column.")
    println(io)
    @printf(io, "%6s %8s %8s %10s %10s %10s %12s %12s\n", "y0", "mu_bar", "eps",
            "k(||g||)", "k(tau)", "ratio", "hprod(g)", "hprod(tau)")
    println(io, "-"^88)
    for y0 in SS_Y0, μ̄ in SS_MUBAR, ε in SS_EPS
        a = sel(regime = :rate, rule = "RGradCapped",    param = μ̄, ε = ε, y0 = y0, x0 = 0.0)
        b = sel(regime = :rate, rule = "RGradCappedTau", param = μ̄, ε = ε, y0 = y0, x0 = 0.0)
        (isempty(a) || isempty(b)) && continue
        ra, rb = a[1], b[1]
        rat = (ra.kesc === nothing || rb.kesc === nothing || rb.kesc == 0) ? "n/a" :
              @sprintf("%.3f", ra.kesc / rb.kesc)
        @printf(io, "%6.0e %8g %8.0e %10s %10s %10s %12d %12d\n", y0, μ̄, ε,
                kesc_s(ra), kesc_s(rb), rat, ra.h_evals, rb.h_evals)
    end
    println(io)
    println(io, "The last two columns are the extra curvature cost of tau. It is")
    println(io, "not free, and the comparison is stated in evaluations rather than")
    println(io, "asserted.")
    save_table(arch, "exp15_measures.txt", String(take!(io)))

    io = IOBuffer()
    println(io, "The three outcomes over the grid, outcome regime, tol = $(SS_TOL_OUTCOME).")
    println(io, "false_convergence: the run met its first-order criterion with")
    println(io, "|y| <= $(SS_THRESH), that is at the saddle. fast: k_esc <= $(SS_FAST).")
    println(io, "never: neither escaped nor converged within $(SS_KMAX) iterations.")
    println(io)
    @printf(io, "%-16s %8s %6s %6s %8s %20s %20s\n", "rule", "param", "y0", "x0",
            "eps", "outcome", "with tol_H = 1e-6")
    println(io, "-"^94)
    for r in rows
        r.regime === :outcome || continue
        r.kind in (:grad, :dfo) || continue
        s2 = [q for q in rows if q.regime === :second_order && q.rule == r.rule &&
              q.param == r.param && q.ε == r.ε && q.y0 == r.y0 && q.x0 == r.x0]
        @printf(io, "%-16s %8g %6.0e %6.1f %8.0e %20s %20s\n", r.rule, r.param,
                r.y0, r.x0, r.ε, string(ss_class(r)),
                isempty(s2) ? "" : string(ss_class(s2[1])) * " (" *
                                   string(s2[1].status) * ")")
    end
    save_table(arch, "exp15_outcomes.txt", String(take!(io)))
    return nothing
end

# -----------------------------------------------------------------------------
# Figures. Black plus one accent, separated by dash pattern and marker too.
# -----------------------------------------------------------------------------
const SS_BLACK  = RGB(0.0, 0.0, 0.0)
const SS_ACCENT = RGB(0.85, 0.33, 0.10)
const SS_GREY   = RGB(0.60, 0.60, 0.60)

function ss_figures(arch, rows)
    sel(; kw...) = [r for r in rows if all(getfield(r, k) == v for (k, v) in kw)]
    dash = (:solid, :dash, :dot, :dashdot)

    # 1. |y_k| against k for three eps at fixed mu_bar, with the comparator
    let μ̄ = 1.0, y0 = 1e-3
        plt = plot(xlabel = "k", ylabel = "|y_k|", yscale = :log10,
                   legend = :bottomright, size = (640, 430), xscale = :log10)
        for (j, ε) in enumerate([1e-1, 1e-2, 1e-3])
            rs = sel(regime = :rate, rule = "RGradCapped", param = μ̄, ε = ε,
                     y0 = y0, x0 = 0.0)
            isempty(rs) && continue
            ys = abs.(rs[1].ys)
            plot!(plt, 1:length(ys), max.(ys, 1e-18); label = "R-grad, ε = $ε",
                  color = isodd(j) ? SS_BLACK : SS_ACCENT, linestyle = dash[j], lw = 1.7)
            rc = sel(regime = :rate, rule = "RDelta", ε = ε, y0 = y0, x0 = 0.0)
            isempty(rc) || plot!(plt, 1:length(rc[1].ys), max.(abs.(rc[1].ys), 1e-18);
                                 label = j == 1 ? "R-delta comparator" : "",
                                 color = SS_GREY, linestyle = :solid, lw = 1.2)
        end
        hline!(plt, [SS_THRESH]; color = SS_BLACK, linestyle = :dot, lw = 1.5,
               label = "escape threshold")
        savefig_archived(arch, "exp15_fig1_y_vs_k.pdf", plt)
    end

    # 2. k_esc against eps, one series per mu_bar, predicted slope -1
    let y0 = 1e-3
        plt = plot(xlabel = "ε", ylabel = "k_esc", xscale = :log10, yscale = :log10,
                   legend = :topright, size = (620, 430))
        for (j, μ̄) in enumerate(SS_MUBAR)
            E = Float64[]; K = Float64[]
            for ε in SS_EPS
                rs = sel(regime = :rate, rule = "RGradCapped", param = μ̄, ε = ε,
                         y0 = y0, x0 = 0.0)
                (isempty(rs) || rs[1].kesc === nothing || rs[1].kesc == 0) && continue
                push!(E, ε); push!(K, rs[1].kesc)
            end
            isempty(E) && continue
            plot!(plt, E, K; label = "μ̄ = $μ̄", color = isodd(j) ? SS_BLACK : SS_ACCENT,
                  linestyle = dash[mod1(j, 4)], marker = :circle, ms = 3, lw = 1.6)
        end
        let e = [minimum(SS_EPS), maximum(SS_EPS)]
            plot!(plt, e, 1 ./ e .* 1e-2; label = "slope −1", color = SS_GREY,
                  linestyle = :dot, lw = 2)
        end
        savefig_archived(arch, "exp15_fig2_kesc_vs_eps.pdf", plt)
    end

    # 3. the collapse: C*eps*k_esc against eps, one panel per y0
    let panels = []
        for y0 in SS_Y0
            p = plot(xlabel = "ε", ylabel = "C ε k_esc", xscale = :log10,
                     legend = :topright, title = "y₀ = $y0", size = (620, 430))
            for (j, μ̄) in enumerate(SS_MUBAR)
                E = Float64[]; V = Float64[]
                for ε in SS_EPS
                    rs = sel(regime = :rate, rule = "RGradCapped", param = μ̄, ε = ε,
                             y0 = y0, x0 = 0.0)
                    (isempty(rs) || rs[1].kesc === nothing) && continue
                    push!(E, ε); push!(V, μ̄*ε*rs[1].kesc)
                end
                isempty(E) && continue
                plot!(p, E, V; label = "μ̄ = $μ̄", color = isodd(j) ? SS_BLACK : SS_ACCENT,
                      linestyle = dash[mod1(j, 4)], marker = :circle, ms = 3, lw = 1.6)
            end
            hline!(p, [log(1/(2y0))]; color = SS_GREY, linestyle = :dot, lw = 2,
                   label = "ln(1/(2y₀)) = $(round(log(1/(2y0)), digits = 2))")
            push!(panels, p)
        end
        savefig_archived(arch, "exp15_fig3_collapse.pdf",
                         plot(panels...; layout = (1, 2), size = (1180, 430)))
    end

    # 4. the fitted geometric rate against C*eps, line of slope 1
    let plt = plot(xlabel = "ln(1 + C ε)", ylabel = "fitted slope of log|y_k| on k",
                   xscale = :log10, yscale = :log10, legend = :bottomright,
                   size = (620, 430))
        X = Float64[]; Y = Float64[]
        for μ̄ in SS_MUBAR, ε in SS_EPS, y0 in SS_Y0
            rs = sel(regime = :rate, rule = "RGradCapped", param = μ̄, ε = ε,
                     y0 = y0, x0 = 0.0)
            (isempty(rs) || !isfinite(rs[1].rate) || rs[1].rate <= 0) && continue
            push!(X, log1p(μ̄*ε)); push!(Y, rs[1].rate)
        end
        isempty(X) || scatter!(plt, X, Y; label = "R-grad cells", color = SS_BLACK,
                               ms = 4, msw = 0)
        if !isempty(X)
            lo, hi = minimum(X), maximum(X)
            plot!(plt, [lo, hi], [lo, hi]; label = "slope 1", color = SS_ACCENT,
                  linestyle = :dash, lw = 2)
        end
        savefig_archived(arch, "exp15_fig4_rate.pdf", plt)
    end

    # 5. the three-outcome map
    let panels = []
        for y0 in SS_Y0
            xs = Float64[]; ys = Float64[]; cs = []; ms = []
            for (i, μ̄) in enumerate(SS_MUBAR), (j, ε) in enumerate(SS_EPS)
                rs = sel(regime = :outcome, rule = "RGradCapped", param = μ̄, ε = ε,
                         y0 = y0, x0 = 0.0)
                isempty(rs) && continue
                c = ss_class(rs[1])
                push!(xs, i); push!(ys, j)
                push!(cs, c === :false_convergence ? SS_ACCENT :
                          c === :fast ? SS_GREY : SS_BLACK)
                push!(ms, c === :false_convergence ? :xcross :
                          c === :fast ? :circle : :square)
            end
            p = plot(xlabel = "μ̄", ylabel = "ε", title = "y₀ = $y0",
                     xticks = (1:length(SS_MUBAR), string.(SS_MUBAR)),
                     yticks = (1:length(SS_EPS), string.(SS_EPS)), legend = false,
                     size = (560, 430))
            for i in eachindex(xs)
                scatter!(p, [xs[i]], [ys[i]]; color = cs[i], marker = ms[i], ms = 8,
                         msw = 1)
            end
            push!(panels, p)
        end
        savefig_archived(arch, "exp15_fig5_outcomes.pdf",
                         plot(panels...; layout = (1, 2), size = (1120, 430)))
    end

    # 6. ||g|| against tau on one panel: geometric against arithmetic
    let μ̄ = 1.0, ε = 1e-2, y0 = 1e-3
        plt = plot(xlabel = "k", ylabel = "|y_k|", yscale = :log10,
                   legend = :bottomright, size = (640, 430))
        for (j, (nm, lab)) in enumerate((("RGradCapped", "ω = ‖g‖, geometric"),
                                         ("RGradCappedTau", "ω = τ, arithmetic")))
            rs = sel(regime = :rate, rule = nm, param = μ̄, ε = ε, y0 = y0, x0 = 0.0)
            isempty(rs) && continue
            ys = abs.(rs[1].ys)
            plot!(plt, 0:length(ys)-1, max.(ys, 1e-18); label = lab,
                  color = j == 1 ? SS_BLACK : SS_ACCENT,
                  linestyle = j == 1 ? :solid : :dash, lw = 1.8)
        end
        hline!(plt, [SS_THRESH]; color = SS_GREY, linestyle = :dot, lw = 1.5,
               label = "escape threshold")
        savefig_archived(arch, "exp15_fig6_measures.pdf", plt)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    slow_saddle()
end
