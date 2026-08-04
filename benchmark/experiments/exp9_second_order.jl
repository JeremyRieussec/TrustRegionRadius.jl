# =============================================================================
# benchmark/experiments/exp9_second_order.jl
#
# EXPERIMENT 9 -- first- vs second-order anchoring.
#
# Three claims, one script.
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
#   julia --project=benchmark benchmark/experiments/exp9_second_order.jl
# =============================================================================

using Plots
include(joinpath(@__DIR__, "..", "archive.jl"))
include(joinpath(@__DIR__, "..", "harness.jl"))
include(joinpath(@__DIR__, "..", "config.jl"))

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

"Where the run stopped: the saddle, a minimiser, or neither."
function limit_point(x)
    norm(x .- SADDLE) < 1e-5              && return "saddle"
    abs(abs(x[1]) - MINIMISER) < 1e-5 &&
        abs(x[2]) < 1e-5                  && return "minimiser"
    return "other"
end

# -----------------------------------------------------------------------------
# (a) and (c): the grid
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

function run_grid()
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

function grid_table(rows)
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
    return String(take!(io))
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
    r = rows[findfirst(x -> !isempty(x.τ), rows)]
    n = length(r.g)
    plot!(plt, 0:(n - 1), max.(r.g, 1e-300);        label = "‖g_k‖", ls = :dash)
    plot!(plt, 0:(length(r.τ) - 1), max.(r.τ, 1e-300); label = "τ_k")
    return plt
end

"λ_min(B_k): negative exactly where the first-order measure is misleading."
function plot_curvature(rows)
    r = rows[findfirst(x -> !isempty(x.λ), rows)]
    plt = plot(0:(length(r.λ) - 1), r.λ; xlabel = "iteration k",
               ylabel = "λ_min(B_k)", title = "model curvature along the run",
               label = r.label, lw = 2, legend = :bottomright)
    hline!(plt, [0.0]; ls = :dot, c = :black, label = "")
    return plt
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function main()
    arch = ExperimentArchive(tag = "second_order")
    save_config(arch; rules = ["see CONFIGS"], params = "tol=$TOL, tol_H=$TOL_H",
                extra = Dict("experiment" => "exp9_second_order",
                             "problem" => "x⁴/4 − x²/2 + y²/2 from (0, 0.7)"))

    rows = run_grid()
    save_table(arch, "exp9_grid.txt", grid_table(rows))
    print(grid_table(rows))

    savefig_archived(arch, "exp9_radius.pdf",      plot_radius(rows))
    savefig_archived(arch, "exp9_measure_gap.pdf", plot_measure_gap(rows))
    savefig_archived(arch, "exp9_curvature.pdf",   plot_curvature(rows))

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

        The grid separates the two ingredients. τ with plain Steihaug keeps the
        radius alive but has no reliable negative-curvature step, and EigenPoint
        with ‖g‖-anchoring has the step but no radius to take it in. Only the
        combination works, which is the content of the claim that the measure and
        the subsolver are independent requirements.
        """)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
