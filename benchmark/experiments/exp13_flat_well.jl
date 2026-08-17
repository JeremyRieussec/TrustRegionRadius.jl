# =============================================================================
# benchmark/experiments/exp13_flat_well.jl
#
# EXPERIMENT 13 -- the radius thresholds, measured rather than estimated.
#
# On the family P2 of the manuscript,
#
#     f(x,y) = ½x² + ε(¼y⁴ − ½y²),   ε ∈ (0, ½),
#
# the minimisers sit at (0, ±1) with ∇²f = diag(1, 2ε), so λ* = 2ε and the three
# thresholds of `eqn: three thresholds` are available in closed form:
#
#     RDFO trapped below   (1−γ2)/(λ*(γ3+1−γ2)) = 0.1/ε   at γ2 = ½, γ3 = 2
#     exact                1/λ*                 = 1/(2ε)
#     κ̄ = 8/λ*                                  = 4/ε
#
# ε moves the curvature at the solution without moving the solution, so a
# bisection on ε reads the thresholds off directly. That is the whole design.
#
# WHAT IS CLAIMED
#
#   (a) RGradCapped(μ_max = μ̄) is trapped for every ε < 1/(2μ̄), and the critical
#       ε is exactly 1/(2μ̄). θ_k = μ_k ≤ μ̄ identically, so the constraint binds
#       at every late iteration, with no dependence on Δ0 at all: initial_radius
#       for this rule ignores Δ0 and returns μ·‖g‖.
#
#   (b) RDFO(ζ) is trapped for ε ≤ 1/(10ζ). The active iteration maps
#       θ ↦ γ(θ)θ/(1 − λ*θ), with γ = γ3 below ζ and γ2 above, and that map
#       leaves [0, θ†] invariant, θ† = (1−γ2)/λ*, exactly when
#       ζ ≤ (1−γ2)/(λ*(γ3+1−γ2)). The condition is SUFFICIENT, so a measured
#       critical ε below the prediction is consistent with the theory and one
#       above it refutes the proposition.
#
#   (c) RGrad uncapped escapes on every ε, after at most ⌈log_{γ3}(1/(λ*μ0))⌉
#       expansions. This is the control arm and appears on every row.
#
# WHAT WOULD FALSIFY IT
#
#   A measured critical ε ABOVE the prediction for any RDFO(ζ) row, since the
#   condition is sufficient. For RGradCapped the condition is necessary and
#   sufficient, so a measured value on either side of 1/(2μ̄) is a finding.
#   An `inactivity_index` of `nothing` on any RGrad row contradicts the one
#   unconditional result in the survey.
#
#   julia --project=benchmark -e 'include("benchmark/initialisation.jl"); flat_well_experiment()'
#
# Format note: no `using` and no `include` here. `initialisation.jl` loads the
# packages, `harness.jl`, `archive.jl` and `config.jl` once, in order. Including
# them per file re-evaluates `config.jl`'s `const`s, which Julia rejects.
#
# That format rule is why this file cannot be run as
# `julia --project=benchmark benchmark/experiments/exp13_flat_well.jl`: with no
# `using` of its own it has no packages when run directly. The `PROGRAM_FILE`
# guard at the foot matches exp9 and the rest, and is reached only when the file
# is executed after the environment is already loaded. Go through
# `initialisation.jl`, as the line above does.
#
# -----------------------------------------------------------------------------
# The clean prefix, and why every statistic here is computed over it
#
# A trapped run does not terminate. It converges linearly, and once ‖g_k‖ reaches
# the scale at which f(x_k) − f(x_k + s_k) cancels, `ared` underflows, ρ_k drops
# below η1, the multiplier or the radius collapses to its floor, and the
# iteration goes on taking steps of negligible length. Those steps still satisfy
# ‖s_k‖ = Δ_k, so they are recorded as ACTIVE. `active_fraction` then returns 1.0
# and `inactivity_index` returns `nothing`: the right answer for the wrong
# reason, and indistinguishable from a genuinely trapped run unless the
# arithmetic is checked.
#
# So every activity statistic below is computed over `clean_prefix`, and both
# lengths are reported in every table.
# =============================================================================

const FW_GAMMA = (γ1 = 0.25, γ2 = 0.5, γ3 = 2.0)
const FW_ETA   = (η = 0.1, η1 = 0.1, η2 = 0.9)
const FW_TOL   = 1e-10
const FW_KMAX  = 3000
const FW_X0    = [0.0, 1.2]
const FW_XREF  = [0.0, 1.0]

const FW_ZETAS = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
const FW_MUBAR = [50.0, 100.0, 500.0, 1000.0]
const FW_BRACKET = (1e-7, 0.49)
const FW_BISECT_STEPS = 45

"""
    clean_prefix(stats; ρlo = 0.9, ρhi = 1.1, gmin = 1e-12) -> Int

Index of the last iteration before the arithmetic degrades: the first `k` at
which `ρ_k` leaves `[ρlo, ρhi]` or `‖g_k‖` falls below `gmin`, minus one.

Returns `0` when the very first iteration is already degraded, and the full
per-iteration length when none is.

Alignment: the state trajectories carry one entry more than the per-iteration
ones and they agree at the head, so `‖g_k‖` for the iteration that produced
`ρ_k` is `g[k]`, and the pairing is with `g[1:end-1]`.
"""
function clean_prefix(stats; ρlo = 0.9, ρhi = 1.1, gmin = 1e-12)
    ρ = get(stats.solver_specific, :ratio_trajectory, Float64[])
    g = get(stats.solver_specific, :grad_trajectory,  Float64[])
    n = min(length(ρ), max(length(g) - 1, 0))
    for k in 1:n
        (isfinite(ρ[k]) && ρlo <= ρ[k] <= ρhi && g[k] > gmin) || return k - 1
    end
    return n
end

"Activity over the clean prefix only, so a collapsed run cannot read as trapped."
function clean_activity(stats)
    m = clean_prefix(stats)
    a = get(stats.solver_specific, :active_trajectory, Bool[])
    m == 0 && return (frac = NaN, still_active = false, m = 0, total = length(a))
    pre = a[1:m]
    return (frac = count(pre) / m, still_active = pre[end], m = m, total = length(a))
end

"One run of the flat well, with everything the experiment needs recorded."
function fw_run(rule, ε; x0 = FW_X0, kmax = FW_KMAX, Δ0 = nothing)
    nlp = flat_well(ε; x0 = x0)
    g0  = norm(grad(nlp, collect(float.(x0))))
    # Δ0 = ½‖g0‖/φ''(y0), so RDFO starts inside the basin of prop: flat well (ii).
    d0  = Δ0 === nothing ? 0.5 * g0 / flat_well_curvature(ε, x0[2]) : Δ0
    return tr_solve(nlp; rule = rule, model = ExactHessian(), subsolver = SteihaugCG(),
                    params = TRParams(η = FW_ETA.η, η1 = FW_ETA.η1, η2 = FW_ETA.η2,
                                      Δ0 = d0, tol = FW_TOL, max_iterations = kmax),
                    x_ref = FW_XREF, trace = true)
end

mk_rdfo(ζ)   = RDFO(γ1 = FW_GAMMA.γ1, γ2 = FW_GAMMA.γ2, γ3 = FW_GAMMA.γ3, ζ = ζ)
mk_capped(μ) = RGradCapped(γ1 = FW_GAMMA.γ1, γ2 = FW_GAMMA.γ2, γ3 = FW_GAMMA.γ3,
                           μ = min(1.0, μ), μ_max = μ)
mk_grad(μ0)  = RGrad(γ1 = FW_GAMMA.γ1, γ2 = FW_GAMMA.γ2, γ3 = FW_GAMMA.γ3, μ = μ0)

"Whether the constraint is still binding at the end of the clean prefix."
still_trapped(rule_f, ε) = clean_activity(fw_run(rule_f(), ε)).still_active

"""
    critical_epsilon(rule_f; bracket, steps) -> (ε, lo, hi)

Largest `ε` at which the constraint is still binding at the end of the clean
prefix, by geometric bisection. Geometric because the thresholds are reciprocal
in `ε` and span four decades over the parameter grid.

The bracket is reported alongside the estimate: a run that saturates it has not
measured a threshold, it has measured the edge of the search.
"""
function critical_epsilon(rule_f; bracket = FW_BRACKET, steps = FW_BISECT_STEPS)
    lo, hi = bracket
    trapped_lo = still_trapped(rule_f, lo)
    trapped_hi = still_trapped(rule_f, hi)
    trapped_lo || return (NaN, lo, hi)          # not trapped even at the bottom
    trapped_hi && return (hi, lo, hi)           # trapped everywhere: saturated
    for _ in 1:steps
        mid = sqrt(lo * hi)
        still_trapped(rule_f, mid) ? (lo = mid) : (hi = mid)
    end
    return (lo, bracket[1], bracket[2])
end

"""
    trace_table(stats, ε; rows) -> String

Per-iteration table of k, ‖g_k‖, θ_k, 1/φ''(y_k), ρ_k, active and branch.

`θ` comes from `theta_trajectory` and the branch from `:branch_trajectory`, so
nothing here re-derives a quantity the solver already recorded.
"""
function trace_table(stats, ε; rows = 12)
    ss = stats.solver_specific
    θ  = theta_trajectory(stats)
    g  = ss[:grad_trajectory]
    ρ  = ss[:ratio_trajectory]
    a  = ss[:active_trajectory]
    br = ss[:branch_trajectory]
    d  = get(ss, :dist_trajectory, Float64[])
    n  = min(rows, length(ρ))
    io = IOBuffer()
    println(io, @sprintf("%4s %13s %13s %10s %10s %7s %-14s", "k", "|g_k|", "theta_k",
                         "1/phi''", "rho_k", "active", "branch"))
    println(io, "-"^78)
    for k in 1:n
        # y_k from the distance to (0,1) along the invariant line, which the run
        # records: y = 1 + dist, since the iterates approach from above.
        y = k <= length(d) ? 1 + d[k] : NaN
        h = flat_well_curvature(ε, y)
        println(io, @sprintf("%4d %13.4e %13.4e %10.4g %10.4f %7s %-14s",
                             k - 1, g[k], θ[k], 1 / h, ρ[k],
                             a[k] ? "yes" : "no", string(br[k])))
    end
    return String(take!(io))
end

"""
    flat_well_experiment(; save = true) -> NamedTuple

Bisect for the critical ε of every rule and parameter, run the uncapped control
on every row, write the two tables and the figure, and archive the lot.
"""
function flat_well_experiment(; save::Bool = true)
    arch = save ? ExperimentArchive(tag = "flat_well") : nothing

    rows = NamedTuple[]
    println("Bisecting for the critical eps. Bracket ", FW_BRACKET,
            ", ", FW_BISECT_STEPS, " geometric steps.\n")
    @printf("%-22s %12s %12s %8s %9s %9s %7s\n",
            "configuration", "eps pred", "eps meas", "ratio", "clean", "total", "k* ctrl")
    println("-"^88)

    for ζ in FW_ZETAS
        pred = 1 / (10ζ)
        meas, blo, bhi = critical_epsilon(() -> mk_rdfo(ζ))
        st  = fw_run(mk_rdfo(ζ), isfinite(meas) ? meas : pred)
        act = clean_activity(st)
        ctl = inactivity_index(fw_run(mk_grad(1.0), isfinite(meas) ? meas : pred))
        push!(rows, (rule = "RDFO", param = ζ, pred = pred, meas = meas,
                     clean = act.m, total = act.total, kstar = ctl,
                     lo = blo, hi = bhi))
        @printf("%-22s %12.6g %12.6g %8.4f %9d %9d %7s\n",
                "RDFO(zeta=$ζ)", pred, meas, meas / pred, act.m, act.total,
                ctl === nothing ? "NONE" : string(ctl))
    end

    for μ in FW_MUBAR
        pred = 1 / (2μ)
        meas, blo, bhi = critical_epsilon(() -> mk_capped(μ))
        st  = fw_run(mk_capped(μ), isfinite(meas) ? meas : pred)
        act = clean_activity(st)
        ctl = inactivity_index(fw_run(mk_grad(1.0), isfinite(meas) ? meas : pred))
        push!(rows, (rule = "RGradCapped", param = μ, pred = pred, meas = meas,
                     clean = act.m, total = act.total, kstar = ctl,
                     lo = blo, hi = bhi))
        @printf("%-22s %12.6g %12.6g %8.4f %9d %9d %7s\n",
                "RGradCapped(mu=$μ)", pred, meas, meas / pred, act.m, act.total,
                ctl === nothing ? "NONE" : string(ctl))
    end

    # ---- the threshold table ------------------------------------------------
    io = IOBuffer()
    println(io, @sprintf("%-22s %12s %12s %8s %9s %9s %8s",
                         "configuration", "eps pred", "eps meas", "ratio",
                         "clean", "total", "k* ctrl"))
    println(io, "-"^86)
    for r in rows
        println(io, @sprintf("%-22s %12.6g %12.6g %8.4f %9d %9d %8s",
                             r.rule * "(" * string(r.param) * ")", r.pred, r.meas,
                             r.meas / r.pred, r.clean, r.total,
                             r.kstar === nothing ? "NONE" : string(r.kstar)))
    end
    println(io)
    println(io, "eps pred: 1/(10 zeta) for RDFO, 1/(2 mu_max) for RGradCapped.")
    println(io, "ratio > 1 on an RDFO row would refute the proposition, whose")
    println(io, "condition is sufficient. For RGradCapped it is exact, so either")
    println(io, "side is a finding. k* is inactivity_index for uncapped RGrad on")
    println(io, "the same eps, and must be a finite integer on every row.")
    println(io, "clean/total: iterations before the arithmetic degrades, against")
    println(io, "the whole run. Activity is measured over the clean prefix only.")
    table = String(take!(io))

    # ---- the two traces -----------------------------------------------------
    # Both traces start at y0 = 2. The specification gives y0 = 2 only for the
    # escaping run, but the trapped run's stated expectations (1/phi'' starting
    # at 0.36, inactive for k = 0 and 1, active from k = 2) are the y0 = 2
    # numbers: at the default y0 = 1.2 the run is active from k = 0 and 1/phi''
    # starts at 1.205, because 1/phi''(2) = 1/(0.25*11) = 0.3636 and
    # 1/phi''(1.2) = 1/(0.25*3.32) = 1.205.
    trapped_st = fw_run(mk_capped(1.0), 0.25; x0 = [0.0, 2.0])
    escape_st  = fw_run(mk_rdfo(1.0), 0.05; x0 = [0.0, 2.0])
    io2 = IOBuffer()
    println(io2, "TRAPPED: RGradCapped(mu_max = 1), eps = 0.25, x0 = (0, 2)")
    println(io2, "Expected: theta = 1 exactly (multiplier pinned at the cap),")
    println(io2, "1/phi'' rising to 1/(2eps) = 2, active from k = 2, rho ~ 1,")
    println(io2, "and |g| halving each step, the rate identity at lambda* theta = 0.5.")
    println(io2)
    print(io2, trace_table(trapped_st, 0.25))
    println(io2)
    println(io2, "ESCAPING: RDFO(zeta = 1), eps = 0.05, x0 = (0, 2)")
    println(io2, "Expected: one expansion at k = 0 carries theta above the")
    println(io2, "threshold 1/lambda* = 10 and the constraint never binds again.")
    println(io2)
    print(io2, trace_table(escape_st, 0.05))
    traces = String(take!(io2))

    println("\n" * traces)

    # ---- the figure ---------------------------------------------------------
    fig = _flat_well_figure(trapped_st, 0.25, escape_st, 0.05)

    if save
        save_config(arch; rules = ["RDFO", "RGradCapped", "RGrad"],
                    params = TRParams(η = FW_ETA.η, η1 = FW_ETA.η1, η2 = FW_ETA.η2,
                                      tol = FW_TOL, max_iterations = FW_KMAX),
                    extra = Dict(
                        "zeta_grid"      => FW_ZETAS,
                        "mu_max_grid"    => FW_MUBAR,
                        "eps_bracket_lo" => FW_BRACKET[1],
                        "eps_bracket_hi" => FW_BRACKET[2],
                        "bisection_steps"=> FW_BISECT_STEPS,
                        "gamma1" => FW_GAMMA.γ1, "gamma2" => FW_GAMMA.γ2,
                        "gamma3" => FW_GAMMA.γ3,
                        "x0" => FW_X0, "x_ref" => FW_XREF,
                        "clean_prefix_rho_lo" => 0.9, "clean_prefix_rho_hi" => 1.1,
                        "clean_prefix_g_min"  => 1e-12))
        save_table(arch, "exp13_thresholds.txt", table)
        save_table(arch, "exp13_trapped_trace.txt", traces)
        savefig_archived(arch, "fig13_flat_well.pdf", fig)
        println("\nArchived to ", arch.dir)
    end

    return (rows = rows, table = table, traces = traces,
            trapped = trapped_st, escaping = escape_st, archive = arch)
end

"Two panels of θ_k against k, with the 1/λ* threshold drawn across both."
function _flat_well_figure(trapped_st, ε_t, escape_st, ε_e)
    θt = theta_trajectory(trapped_st); θe = theta_trajectory(escape_st)
    at = trapped_st.solver_specific[:active_trajectory]
    ae = escape_st.solver_specific[:active_trajectory]
    nt = min(length(θt), 40); ne = min(length(θe), 12)
    thr_t = 1 / (2ε_t); thr_e = 1 / (2ε_e)

    p1 = Plots.plot(0:nt-1, max.(θt[1:nt], 1e-16); yscale = :log10,
                    label = "θ_k", lw = 2, color = :black, marker = :circle, ms = 3,
                    xlabel = "k", ylabel = "θ_k = Δ_k/‖g_k‖",
                    title = "trapped: RGradCapped(μ̄ = 1), ε = $ε_t")
    Plots.hline!(p1, [thr_t]; label = "1/λ* = $(round(thr_t, digits=3))",
                 lw = 3, color = :red, ls = :dash)

    p2 = Plots.plot(0:ne-1, max.(θe[1:ne], 1e-16); yscale = :log10,
                    label = "θ_k", lw = 2, color = :black, marker = :circle, ms = 3,
                    xlabel = "k", ylabel = "θ_k = Δ_k/‖g_k‖",
                    title = "escaping: RDFO(ζ = 1), ε = $ε_e")
    Plots.hline!(p2, [thr_e]; label = "1/λ* = $(round(thr_e, digits=3))",
                 lw = 3, color = :red, ls = :dash)
    kesc = findfirst(!, ae)
    kesc === nothing || Plots.vline!(p2, [kesc - 1];
        label = "constraint goes inactive, k = $(kesc-1)", lw = 2, color = :blue, ls = :dot)

    return Plots.plot(p1, p2; layout = (1, 2), size = (1100, 430),
                      legend = :bottomright, left_margin = 5Plots.mm,
                      bottom_margin = 5Plots.mm)
end

if abspath(PROGRAM_FILE) == @__FILE__
    flat_well_experiment()
end
