# =============================================================================
# benchmark/experiments/exp9_second_order.jl
#
# EXPERIMENT 9 -- first- vs second-order anchoring.
#
# Four claims, one script.
#
#   (a) THE ANCHOR COLLAPSE.  Started so that the trajectory runs into a saddle,
#       a ‖g‖-anchored rule reports the radius it would use at a solution.
#       RGrad sets Δ = μ‖g‖ → 0 and halts; RDFO finds Δ > ζ‖g‖ = 0 at every
#       iteration and contracts geometrically. Under τ = max{‖g‖, −λ_min} both
#       hold the radius at the scale of the available curvature and escape.
#
#   (b) THE MEASURE GAP.  τ_k / ‖g_k‖ over a run says where the two disagree.
#       It is 1 wherever the model is convex and blows up near a saddle, which
#       is exactly where a first-order diagnostic reports success.
#
#   (c) THE SUBSOLVER IS HALF THE STORY.  τ keeps the radius positive; it does
#       not make the step go anywhere. The 2×2 grid of measure × subsolver shows
#       that only τ *and* EigenPoint together reach a minimiser.
#
#   (d) THE MODEL IS THE OTHER HALF.  Over a model that is positive
#       semidefinite by construction, λ_min ≥ 0 identically, so τ ≡ ‖g‖ and the
#       whole apparatus is an expensive no-op -- which will still report
#       `:second_order` at a saddle.
#
#   julia --project=benchmark benchmark/experiments/exp9_second_order.jl
#
# Format note: no `using` and no `include` here. `initialisation.jl` loads the
# packages, `harness.jl`, `archive.jl` and `config.jl` once, in order, exactly as
# it does for experiments 1-7. Re-including them per file re-ran `config.jl` and
# so re-evaluated its `const`s, which Julia either warns about or rejects.
#
# ---------------------------------------------------------------------------
# Integration note
#
# Part (e) now runs through `run_experiment` rather than a private loop, so it
# archives, resumes and produces `RunRecord`s like every other experiment. That
# needed two things from the harness, both missing, and both mattering here more
# than anywhere else:
#
#   * `solved` counts `:second_order`. It used to test `=== :first_order`, so a
#     τ-run that certified a minimiser was recorded as a FAILURE -- the columns
#     doing best would have shown zero reliability, and the experiment would have
#     concluded the exact opposite of the truth.
#   * `RunRecord` carries `tau_traj` and `lambda_traj`. They were dropped, so
#     claim (b) could not be plotted from an archived run at all.
#
# Parts (a)-(d) keep a direct `tr_solve` path: each is a statement about a single
# trajectory on one hand-built problem, and routing it through the record layer
# would add machinery without adding information.
# =============================================================================

# f = x⁴/4 − x²/2 + y²/2: a saddle at the origin with λ_min = −1, minimisers at
# (±1, 0). The x-axis is invariant under the gradient flow (∂f/∂x = 0 when x = 0),
# so a start on the y-axis runs exactly into the saddle -- no tuning required to
# make the failure appear.
saddle_f(v) = v[1]^4 / 4 - v[1]^2 / 2 + v[2]^2 / 2

const X0        = [0.0, 0.7]
const SADDLE    = [0.0, 0.0]
const MINIMISER = 1.0                       # |x| at either minimiser
const TOL, TOL_H = 1e-8, 1e-6

make_nlp() = ADNLPModel(saddle_f, copy(X0), name = "saddle_quartic")

# A second saddle in higher dimension, so claim (a) is not an artefact of one
# hand-built quartic: the monkey saddle x³ − 3xy² + z²/2, whose origin is a
# degenerate critical point with a cubic escape direction.
monkey_f(v) = v[1]^3 - 3v[1] * v[2]^2 + v[3]^2 / 2

const SADDLE_PROBLEMS = [
    ("saddle_quartic", () -> ADNLPModel(saddle_f, [0.0, 0.7],      name = "saddle_quartic")),
    ("monkey_saddle",  () -> ADNLPModel(monkey_f, [0.05, 0.0, 0.6], name = "monkey_saddle")),
]

"Where the run stopped: the saddle, a minimiser, or neither."
function limit_point(x)
    norm(x .- SADDLE) < 1e-5              && return "saddle"
    abs(abs(x[1]) - MINIMISER) < 1e-5 &&
        abs(x[2]) < 1e-5                  && return "minimiser"
    return "other"
end

# -----------------------------------------------------------------------------
# (a) and (c): measure × subsolver, on the quartic
# -----------------------------------------------------------------------------

const CONFIGS = [
    ("RGrad  ‖g‖  Steihaug",  () -> RGrad(μ = 1.0),      () -> SteihaugCG(),            -1.0),
    ("RGrad  ‖g‖  EigenPoint",() -> RGrad(μ = 1.0),      () -> EigenPoint(SteihaugCG()),-1.0),
    ("RGrad  τ    Steihaug",  () -> RGradTau(μ = 1.0),   () -> SteihaugCG(),            TOL_H),
    ("RGrad  τ    EigenPoint",() -> RGradTau(μ = 1.0),   () -> EigenPoint(SteihaugCG()),TOL_H),
    ("RDFO   ‖g‖  Steihaug",  () -> RDFO(ζ = 1.0),       () -> SteihaugCG(),            -1.0),
    ("RDFO   ‖g‖  EigenPoint",() -> RDFO(ζ = 1.0),       () -> EigenPoint(SteihaugCG()),-1.0),
    ("RDFO   τ    Steihaug",  () -> RDFOTau(ζ = 1.0),    () -> SteihaugCG(),            TOL_H),
    ("RDFO   τ    EigenPoint",() -> RDFOTau(ζ = 1.0),    () -> EigenPoint(SteihaugCG()),TOL_H),
]

function anchor_grid()
    rows = NamedTuple[]
    for (label, rulef, subf, tolH) in CONFIGS
        st = tr_solve(make_nlp(); rule = rulef(), model = ExactHessian(),
                      subsolver = subf(), trace = true,
                      params = TRParams(tol = TOL, tol_H = tolH,
                                        max_iterations = 5_000))
        ss = st.solver_specific
        push!(rows, (label = label, status = st.status, iters = st.iter,
                     limit = limit_point(st.solution), obj = st.objective,
                     Δend = ss[:delta_trajectory][end],
                     τ = get(ss, :tau_trajectory, Float64[]),
                     g = ss[:grad_trajectory],
                     Δ = ss[:delta_trajectory],
                     λ = get(ss, :lambda_min_trajectory, Float64[])))
    end
    return rows
end

function anchor_table(rows)
    io = IOBuffer()
    @printf(io, "%-26s %14s %7s %11s %12s %12s\n",
            "configuration", "status", "iters", "limit", "f(x*)", "Δ_end")
    println(io, "-"^88)
    for r in rows
        @printf(io, "%-26s %14s %7d %11s %12.6f %12.3e\n",
                r.label, string(r.status), r.iters, r.limit, r.obj, r.Δend)
    end
    println(io)
    println(io, "The saddle has f = 0; each minimiser has f = −0.25.")
    println(io, "A run that reports :first_order at the saddle has not failed by its")
    println(io, "own test — ‖g‖ really is zero there. It is the test that is wrong.")
    println(io, "A run reporting :second_order has certified λ_min(B) ≥ −tol_H, which")
    println(io, "is a statement about the MODEL: see exp9_blind.txt.")
    return String(take!(io))
end

# -----------------------------------------------------------------------------
# (d) the blind spot: the same τ-rule over a model that cannot see curvature
# -----------------------------------------------------------------------------

"""
    blind_table()

τ-anchoring over models that are positive semidefinite by construction.

`LBFGSModel` enforces `B ≻ 0`, so `λ_min > 0` always, `τ = max{‖g‖, −λ_min} ≡ ‖g‖`,
and the second-order variant is an expensive no-op -- which will nonetheless
report `:second_order` at a saddle, because the test it passes is a statement
about `B` rather than about `∇²f`.

The cheapest demonstration in the suite that a status is only as strong as the
model behind it, and the reason `reports_negative_curvature` is a trait and the
solver warns when the pairing is made.
"""
function blind_table()
    io = IOBuffer()
    @printf(io, "%-18s %14s %7s %11s %16s %12s\n",
            "model", "status", "iters", "limit", "min λ_min seen", "certified?")
    println(io, "-"^84)
    for (mname, mf) in (("ExactHessian", () -> ExactHessian()),
                        ("SR1Model",     () -> SR1Model(mem = 5)),
                        ("LBFGSModel",   () -> LBFGSModel(mem = 5)))
        st = tr_solve(make_nlp(); rule = RGradTau(μ = 1.0), model = mf(),
                      subsolver = EigenPoint(SteihaugCG()), trace = true,
                      params = TRParams(tol = TOL, tol_H = TOL_H,
                                        max_iterations = 5_000))
        λ = get(st.solver_specific, :lambda_min_trajectory, Float64[])
        @printf(io, "%-18s %14s %7d %11s %16s %12s\n",
                mname, string(st.status), st.iter, limit_point(st.solution),
                isempty(λ) ? "--" : @sprintf("%.4f", minimum(λ)),
                st.status === :second_order ? "yes" : "no")
    end
    println(io)
    println(io, "LBFGSModel never reports λ_min < 0, so τ ≡ ‖g‖ and the τ-rule is")
    println(io, "identical to its ‖g‖-twin. If it still reports :second_order, the")
    println(io, "status certifies the model, not the function.")
    return String(take!(io))
end

# -----------------------------------------------------------------------------
# (e) the paired sweep, through the shared harness
# -----------------------------------------------------------------------------

"""
    paired_sweep(arch) -> Vector{RunRecord}

Every ‖g‖-anchored rule beside its τ-twin, on the saddle problems, through
`run_experiment` -- so the runs are archived and resumable like every other
experiment.

The table to read is `success_table(...; second_order = true)`: its last column
counts runs that stopped with a *certified* `:second_order`. A configuration
solving every problem while certifying none has been running a first-order method
under a second-order name.
"""
function paired_sweep(arch)
    configs = paired_configs()          # model = ExactHessian, sub = EigenPoint
    records = run_experiment(SADDLE_PROBLEMS, configs;
                             params = SECOND_ORDER_PARAMS, trace = true,
                             archive = arch)
    save_table(arch, "exp9_paired.txt",
               success_table(records, SADDLE_PROBLEMS, configs; second_order = true))
    return records
end

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

"Δ_k for each configuration: the collapse under ‖g‖, the plateau under τ."
function plot_radius(rows)
    plt = plot(; xlabel = "iteration k", ylabel = "Δ_k", yscale = :log10,
                 title = "radius under ‖g‖- and τ-anchoring", legend = :bottomleft, lw = 2)
    for r in rows
        isempty(r.Δ) && continue
        plot!(plt, 0:(length(r.Δ) - 1), max.(r.Δ, 1e-300); label = r.label)
    end
    return plt
end

"τ_k against ‖g_k‖: equal on the convex part, divergent at the saddle."
function plot_measure_gap(rows)
    plt = plot(; xlabel = "iteration k", ylabel = "measure", yscale = :log10,
                 title = "τ_k = max{‖g_k‖, −λ_min} versus ‖g_k‖",
                 legend = :bottomleft, lw = 2)
    i = findfirst(x -> !isempty(x.τ), rows)
    i === nothing && return plt
    r = rows[i]
    n = length(r.g)
    plot!(plt, 0:(n - 1), max.(r.g, 1e-300);           label = "‖g_k‖", ls = :dash)
    plot!(plt, 0:(length(r.τ) - 1), max.(r.τ, 1e-300); label = "τ_k")
    return plt
end

"""
The ratio τ_k/‖g_k‖ on its own axis: 1 on the convex part, unbounded at the
saddle. Separate from `plot_measure_gap` because the ratio is the quantity with
the interpretation -- where it exceeds 1, the gradient alone calls the point
critical and the curvature does not.
"""
function plot_gap_ratio(rows)
    plt = plot(; xlabel = "iteration k", ylabel = "τ_k / ‖g_k‖", yscale = :log10,
                 title = "where the two measures disagree", legend = :topleft, lw = 2)
    for r in rows
        (isempty(r.τ) || isempty(r.g)) && continue
        n = min(length(r.τ), length(r.g))
        plot!(plt, 0:(n - 1),
              [max(r.τ[k] / max(r.g[k], 1e-300), 1e-300) for k in 1:n]; label = r.label)
    end
    hline!(plt, [1.0]; ls = :dot, c = :black, label = "τ = ‖g‖")
    return plt
end

"λ_min(B_k): negative exactly where the first-order measure is misleading."
function plot_curvature(rows)
    i = findfirst(x -> !isempty(x.λ), rows)
    i === nothing && return plot(; title = "no curvature recorded")
    r = rows[i]
    plt = plot(0:(length(r.λ) - 1), r.λ; xlabel = "iteration k",
               ylabel = "λ_min(B_k)", title = "model curvature along the run",
               label = r.label, lw = 2, legend = :bottomright)
    hline!(plt, [0.0]; ls = :dot, c = :black, label = "")
    return plt
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function second_order()
    arch = ExperimentArchive(tag = "second_order")
    save_config(arch; rules = [nm for (nm, _) in TAU_RULES],
                # Both axes are varied deliberately here: anchor_grid crosses the
                # rules with the two subsolvers, and blind_table crosses the model
                # with the measure. Recording one fixed pair would misdescribe it.
                models = [("ExactHessian", () -> ExactHessian()),
                          ("SR1Model",     () -> SR1Model(mem = 5)),
                          ("LBFGSModel",   () -> LBFGSModel(mem = 5))],
                subsolvers = [("Steihaug",   () -> SteihaugCG()),
                              ("EigenPoint", () -> EigenPoint(SteihaugCG()))],
                params = "tol=$TOL, tol_H=$TOL_H",
                extra = Dict("experiment" => "exp9_second_order",
                             "problem" => "x⁴/4 − x²/2 + y²/2 from (0, 0.7)"))

    rows = anchor_grid()
    save_table(arch, "exp9_grid.txt", anchor_table(rows))
    print(anchor_table(rows))

    save_table(arch, "exp9_blind.txt", blind_table())
    print(blind_table())

    savefig_archived(arch, "exp9_radius.pdf",      plot_radius(rows))
    savefig_archived(arch, "exp9_measure_gap.pdf", plot_measure_gap(rows))
    savefig_archived(arch, "exp9_gap_ratio.pdf",   plot_gap_ratio(rows))
    savefig_archived(arch, "exp9_curvature.pdf",   plot_curvature(rows))

    paired_sweep(arch)

    finalize_archive(arch; notes = """
        First- versus second-order anchoring on a quartic whose saddle lies on the
        trajectory: f = x⁴/4 − x²/2 + y²/2 started from (0, 0.7). The x-axis is
        invariant under the gradient flow, so the run meets the saddle exactly and
        the failure needs no tuning to provoke.

        Under ‖g‖-anchoring the radius reports the value it would take at a
        solution, because ‖g‖ = 0 at the saddle just as it does at a minimiser.
        RGrad sets Δ = μ‖g‖ = 0 and halts; RDFO finds Δ > ζ‖g‖ at every iteration
        and contracts by γ₂ for ever. Both then report :first_order — correctly by
        their own test, which is the point: the test cannot tell the two apart.

        Under τ = max{‖g‖, −λ_min} the radius stays at the scale of the curvature
        available, and with a subsolver that can move along the negative direction
        the run leaves the saddle and reaches a minimiser, reporting :second_order.

        The grid separates the first two ingredients. τ with plain Steihaug keeps
        the radius alive but has no reliable negative-curvature step, and EigenPoint
        with ‖g‖-anchoring has the step but no radius to take it in. Only the
        combination works.

        exp9_blind.txt adds the third ingredient, the model. LBFGSModel enforces
        B ≻ 0, so λ_min > 0 always and τ ≡ ‖g‖: the τ-rule becomes identical to its
        ‖g‖-twin while still able to report :second_order. The status is a statement
        about the model Hessian, and is only as strong as the model behind it.

        exp9_paired.txt runs every ‖g‖-rule beside its τ-twin through the shared
        harness, on two saddle problems. Read the last column — certified
        :second_order — against the solved count: a configuration solving
        everything and certifying nothing has been running a first-order method
        under a second-order name.
        """)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    second_order()
end
