# =============================================================================
# notebooks/Saddle/saddle_problem.jl
#
# The shared setup for the saddle studies, extracted verbatim from
# `saddle_discussion_v1.ipynb` by `scratchpad/extract_saddle.py` rather than
# retyped, so the two cannot drift. The notebook is the source and is left
# untouched.
#
# Two display blocks were dropped in the copy, the printed table of critical
# points and a single demonstration run. Nothing else was changed, added or
# reordered inside a block. The blocks themselves are ordered constants first.
#
# The caller supplies the packages:
#
#     using TrustRegionRadius, ADNLPModels, LinearAlgebra, Printf
#     include("saddle_problem.jl")
#
# ON THE CONSTANTS. Section~"Settings" of Survey-part3-v1.tex says that every
# number used in the paper is fixed there, and carries a TODO for the table
# rather than the table. There is therefore nothing in the paper to check these
# against. They differ from `benchmark/config.jl` in two places, eta2 (0.75 here,
# 0.9 there) and zeta (0.5 here, 100.0 there). Both files are internally
# consistent, and which set the paper adopts is the author's to settle.
# =============================================================================

using TrustRegionRadius
using ADNLPModels, NLPModels
using LinearAlgebra, Printf, Statistics
using Plots, LaTeXStrings

# `import`, not `using`, and after Plots. Two separate reasons, both checked.
#
# ORDER. PyPlot loads a Python whose Qt6 libraries shadow the ones GR_jll binds
# to, so `using PyPlot` ahead of `using Plots` aborts the include with
#     InitError: could not load library "...Qt6Concurrent.dll"
#     during initialization of module Qt6Base_jll
# Both orders were tried in this environment and only this one loads.
#
# IMPORT. PyPlot and Plots both export `plot`, `savefig`, `contour`, `scatter`
# and more. Under `using` both, every unqualified use in Main becomes ambiguous
# and throws `UndefVarError: plot not defined`, which is what broke
# `mu_cap_gradient_descent_v1.ipynb`. `import` binds the module name alone and
# exports nothing, and `plot_run` already qualifies every PyPlot call it makes,
# so the notebooks keep Plots' unqualified names.
import PyPlot

gr()
default(fontfamily = "Computer Modern", framestyle = :box, grid = true,
        gridalpha = 0.25, legendfontsize = 7, guidefontsize = 9,
        titlefontsize = 10, tickfontsize = 8)

# ---------------------------------------------------------------- constants

# ---- standard parameters (change these and re-run) -----------------------
const ETA1, ETA2 = 0.1, 0.75        # acceptance / very-successful thresholds
const G1, G2, G3 = 0.25, 0.5, 2.0   # gamma_1, gamma_2, gamma_3
const ZETA       = 0.5              # R-DFO criticality threshold
const MU0, DELTA0 = 0.5, 1.0        # initial mu and radius
const MU_BAR     = 2.0              # cap for the bounded-mu variant of R-grad

const TOL, KMAX = 1e-9, 2000

# ------------------------------------------------------- critical points

# ---------------------------------------------------------------- critical points
const ORIGIN = [0.0, 0.0]                                   # degenerate, singular, PSD
const SADDLE = [0.75, 0.0]                                  # nonsingular saddle
const MINP   = [(1+sqrt(5))/4,  sqrt((sqrt(5)-1)/4)]        # local minimum
const MINM   = [(1+sqrt(5))/4, -sqrt((sqrt(5)-1)/4)]        # local minimum

const X0_DEFAULT = [-0.5, 0.6]

const CRITPTS = [("origin",  ORIGIN), ("saddle", SADDLE),
                 ("min+",    MINP),   ("min-",   MINM)]

# The printed table of critical points used to stand here, ahead of the
# definitions of `f`, `grad` and `hess`, so the include died on its first call
# to `hess`. It is `critical_points_table()` below, defined after them and
# called by whoever wants it, which is what the header of this file already
# claimed. Four notebooks include this file and none of them wants the table
# printed at them on load.

# ---------------------------------------------------------------- test function
f(p) = p[1]^4 - p[1]^3 + (0.25 - p[1]/2)*p[2]^2 + p[2]^4/4

grad(p) = [4p[1]^3 - 3p[1]^2 - p[2]^2/2,
           p[2]^3 + 2p[2]*(0.25 - p[1]/2)]

hess(p) = [6p[1]*(2p[1]-1)   -p[2];
           -p[2]             (-p[1] + 3p[2]^2 + 0.5)]

# the same f, as the model the solver actually sees
make_nlp(x0 = X0_DEFAULT) = ADNLPModel(f, collect(float.(x0)), name = "quartic")

"""
    critical_points_table(io = stdout)

Print `f`, `‖grad‖` and the two eigenvalues of the Hessian at each of the four
critical points. A display, called by a notebook that wants it rather than run on
include.
"""
function critical_points_table(io::IO = stdout)
    println(io, "Critical points of f\n")
    @printf(io, "%-8s %-22s %12s %12s %22s\n",
            "name", "x", "f(x)", "||grad||", "eig(hess)")
    println(io, "-"^80)
    for (nm, p) in CRITPTS
        w = eigvals(Symmetric(hess(p)))
        @printf(io, "%-8s (%+.6f, %+.6f) %12.8f %12.2e   (%+.5f, %+.5f)\n",
                nm, p[1], p[2], f(p), norm(grad(p)), w[1], w[2])
    end
    return nothing
end

# =============================================================================
# Step diagnostics
#
# Migrated verbatim from `mu_cap_gradient_descent_v1.ipynb`, where they were
# defined locally, because four notebooks need them. That notebook now takes them
# from here.
# =============================================================================

"""
    step_classes(ss) -> (truncated, inside, krylov)

The three classes of step, as fractions of the iterations that happened.

1. truncated at the boundary, `cg_iters == 1` and the step active: the Hessian
   touched neither the direction nor the length;
2. stopped inside after one CG iteration: the direction is `-g_k` and the Hessian
   sets the length through the exact line search on the model;
3. two or more CG iterations.

Only the first is gradient descent in disguise. `ExactMS` reports
`cg_iters == 0` throughout, so the classes do not apply and `(NaN, NaN, NaN)` is
returned. Read the angle instead.
"""
function step_classes(ss)
    cg, ac = ss[:cg_iters_trajectory], ss[:active_trajectory]
    n = length(cg)
    (n == 0 || all(==(0), cg)) && return (NaN, NaN, NaN)
    return (count(i -> cg[i] == 1 &&  ac[i], 1:n) / n,
            count(i -> cg[i] == 1 && !ac[i], 1:n) / n,
            count(i -> cg[i] >= 2,           1:n) / n)
end

"`sin` of the angle between `Hg` and `g`. Zero exactly when `g` is an eigenvector."
function eig_deviation(x)
    g = grad(x); H = hess(x); Hg = H * g
    ng, nHg = norm(g), norm(Hg)
    (ng == 0 || nHg == 0) && return NaN
    return sqrt(max(0.0, 1 - clamp(abs(dot(Hg, g)) / (nHg * ng), 0.0, 1.0)^2))
end

"""
    z1_ratio(x, Δ) -> Float64

`‖z₁‖ / Δ`, where `z₁` is the first Steihaug iterate, the exact minimiser of the
model along `-g`. Steihaug truncates at the first iteration when this reaches
one, and continues to a second when it does not.

Returns `Inf` when `gᵀHg ≤ 0`, where CG goes straight to the boundary.
"""
function z1_ratio(x, Δ)
    g = grad(x); H = hess(x); gHg = dot(g, H * g)
    gHg <= 0 && return Inf
    return (dot(g, g) / gHg) * norm(g) / Δ
end

"The g-weighted variance of the spectrum of `H(x)`, which sets the ExactMS angle."
function spectrum_variance(x)
    g = grad(x); ng = norm(g)
    ng == 0 && return NaN
    F = eigen(Symmetric(hess(x)))
    w = (transpose(F.vectors) * g) .^ 2 ./ ng^2
    return sum(w .* F.values .^ 2) - sum(w .* F.values)^2
end

"""
    loglog_fit(x, y; ymin = 0.0) -> (slope, inter, r2, n)

Least squares slope of `log10(y)` on `log10(x)`, with intercept, `R²` and the
number of usable points. Returns `NaN` below five usable points, so a fit is never
quoted from a handful of iterations. Report `n` and `r2` beside every slope taken
from this.
"""
function loglog_fit(x, y; ymin = 0.0)
    idx = [i for i in eachindex(y) if isfinite(x[i]) && isfinite(y[i]) &&
                                      x[i] > 0 && y[i] > ymin]
    length(idx) < 5 && return (slope = NaN, inter = NaN, r2 = NaN, n = length(idx))
    X, Y = log10.(x[idx]), log10.(y[idx])
    X̄, Ȳ = mean(X), mean(Y)
    sxx = sum(abs2, X .- X̄)
    sxx == 0 && return (slope = NaN, inter = NaN, r2 = NaN, n = length(idx))
    b = sum((X .- X̄) .* (Y .- Ȳ)) / sxx
    a = Ȳ - b * X̄
    ss = sum(abs2, Y .- Ȳ)
    return (slope = b, inter = a,
            r2 = ss == 0 ? NaN : 1 - sum(abs2, Y .- (a .+ b .* X)) / ss,
            n = length(idx))
end

"Distance to the nearest of the four critical points, and its name."
function nearest_crit(x)
    best, bd = "other", Inf
    for (nm, p) in CRITPTS
        d = norm(x .- p); d < bd && (bd = d; best = nm)
    end
    return (name = best, dist = bd)
end

# ------------------------------------------------------------------ runners

"Run one configuration. Returns the GenericExecutionStats from tr_solve, unchanged."
function run_cfg(; rule, model = ExactHessian(), subsolver = SteihaugCG(),
                   x0 = X0_DEFAULT, tol = TOL, tol_H = -1.0, kmax = KMAX,
                   Δ0 = DELTA0, η = ETA1)
    return tr_solve(make_nlp(x0); rule = rule, model = model, subsolver = subsolver,
                    params = TRParams(η = η, η1 = ETA1, η2 = ETA2, Δ0 = Δ0,
                                      tol = tol, tol_H = tol_H, max_iterations = kmax),
                    trace = true)
end

"Same, plus the iterate path collected through the solver's callback."
function solve_path(; x0 = X0_DEFAULT, kwargs...)
    xs = [collect(float.(x0))]
    st = tr_solve(make_nlp(x0);
                  params = TRParams(η = get(kwargs, :η, ETA1), η1 = ETA1, η2 = ETA2,
                                    Δ0 = get(kwargs, :Δ0, DELTA0),
                                    tol = get(kwargs, :tol, TOL),
                                    tol_H = get(kwargs, :tol_H, -1.0),
                                    max_iterations = get(kwargs, :kmax, KMAX)),
                  rule      = kwargs[:rule],
                  model     = get(kwargs, :model, ExactHessian()),
                  subsolver = get(kwargs, :subsolver, SteihaugCG()),
                  trace = true,
                  callback = (_n, _s, stats) -> push!(xs, copy(stats.solution)))
    return st, xs
end

"""
    which_crit(x; tol = 1e-4) -> String

Classify a limit point by distance to the four known critical points, or
`"other"` when the nearest one is further than `tol`.

One distance computation, `nearest_crit`, serves both. The signature and the
`tol` behaviour are unchanged, so the existing notebooks are unaffected.
"""
function which_crit(x; tol = 1e-4)
    nc = nearest_crit(x)
    return nc.dist < tol ? nc.name : "other"
end

# `tail_active` used to be defined here, hand-rolled, with
#     k0 = max(1, floor(Int, 0.9 * length(a)))
# which is off by one against the package: at n = 41 it takes six points where
# `active_fraction` takes five, because floor(0.9n) is the last index before the
# tail and the tail starts at floor(0.9n) + 1. Use the package, which does the
# slice once and correctly, and takes `stats` rather than `stats.solver_specific`.



"""
    plot_run(st, xs; savepath, rule_name, model_name, solver_name, draw_regions, η, η1, η2)

Four-panel diagnostic for one `tr_solve` run.

`st` is the `GenericExecutionStats` returned by `tr_solve(...; trace = true)` and `xs` the
iterate path collected through its `callback` — exactly what `solve_path` returns. Every
series is read from `st.solver_specific`; nothing is recomputed.

`draw_regions` controls the per-iteration trust-region circles in panel (a): `:auto` draws
them only when there are ≤ 60 accepted steps, `true`/`false` force on/off.
"""
function plot_run(st, xs; savepath = nothing, rule_name = "", model_name = "",
                  solver_name = "", draw_regions = :auto,
                  η = ETA1, η1 = ETA1, η2 = ETA2)
    ss = st.solver_specific
    Δv = ss[:delta_trajectory]          # length K+1
    fv = ss[:obj_trajectory]            # length K+1
    ρv = ss[:ratio_trajectory]          # length K
    sv = ss[:step_trajectory]           # length K
    av = ss[:active_trajectory]         # length K
    cv = ss[:accepted_trajectory]       # length K
    K  = length(ρv)

    P = reduce(hcat, xs)
    naccept = count(cv)

    tag = isempty(rule_name) && isempty(model_name) ? "" :
          ": " * join(filter(!isempty, [model_name, rule_name, solver_name]), " + ")

    fig  = PyPlot.figure(figsize = (13.5, 10.5))
    ax_a = fig.add_subplot(2, 2, 1); ax_b = fig.add_subplot(2, 2, 2)
    ax_c = fig.add_subplot(2, 2, 3); ax_d = fig.add_subplot(2, 2, 4)

    # ---------- (a) phase portrait ----------
    ax = ax_a
    xg = range(-0.95, 1.30, length = 420)
    yg = range(-0.85, 0.90, length = 380)
    Xg = [xi for yi in yg, xi in xg]
    Yg = [yi for yi in yg, xi in xg]
    Zg = @. Xg^4 - Xg^3 + (0.25 - Xg/2)*Yg^2 + Yg^4/4
    lv = sort(vcat(collect(range(-0.13, 0.02, length = 14)),
                   collect(range(0.03, 0.9, length = 10))))
    ax.contour(Xg, Yg, Zg, levels = lv, colors = "0.72", linewidths = 0.6, zorder = 1)
    ax.contour(Xg, Yg, Zg, levels = [f(SADDLE)], colors = "crimson",
               linewidths = 1.3, linestyles = "--", zorder = 2)

    xq = range(-0.9, 1.2, length = 22); yq = range(-0.8, 0.8, length = 18)
    QX = Float64[]; QY = Float64[]; QU = Float64[]; QV = Float64[]
    for xi in xq, yi in yq
        gg = -grad([xi, yi]); n = norm(gg)
        n == 0 && continue
        push!(QX, xi); push!(QY, yi); push!(QU, gg[1]/n); push!(QV, gg[2]/n)
    end
    ax.quiver(QX, QY, QU, QV, color = "0.62", width = 0.0028, scale = 42,
              alpha = 0.75, zorder = 1)

    # Per-iteration trust regions on accepted steps. Step i LEAVES xs[i] with radius Δv[i]:
    # xs has K+1 entries and cv has K, so the index is the step, not the iterate reached.
    show_reg = draw_regions === :auto ? naccept <= 60 : draw_regions
    if show_reg
        for i in 1:K
            cv[i] || continue
            ax.add_patch(PyPlot.matplotlib.patches.Circle((P[1, i], P[2, i]), Δv[i],
                         fill = false, ec = "#1565c0", lw = 0.85, alpha = 0.5, zorder = 3))
        end
    end

    ax.plot([MINP[1], MINM[1]], [MINP[2], MINM[2]], "o", ms = 11, mfc = "#2e7d32",
            mec = "k", mew = 1.2, zorder = 6, label = "local minima")
    ax.plot([ORIGIN[1]], [ORIGIN[2]], "D", ms = 8, mfc = "0.55", mec = "k", mew = 1.1,
            zorder = 6, label = "degenerate crit. pt")
    ax.plot([SADDLE[1]], [SADDLE[2]], "*", ms = 24, mfc = "#d32f2f", mec = "k", mew = 1.3,
            zorder = 7, label = "saddle \$x^*\$")

    stride = max(1, size(P, 2) ÷ 400)          # don't paint thousands of markers
    idx = unique(vcat(1:stride:size(P, 2), size(P, 2)))
    ax.plot(P[1, idx], P[2, idx], "-o", color = "#0d47a1", ms = 5.5, lw = 2.0,
            mfc = "white", mec = "#0d47a1", mew = 1.6, zorder = 8, label = "iterates \$x_k\$")
    ax.plot([P[1, 1]], [P[2, 1]], "s", ms = 10, mfc = "#0d47a1", mec = "k", mew = 1.2,
            zorder = 9, label = "\$x_0\$")
    if st.status === :exception                # was "breakdown": no model exists here
        ax.plot([P[1, end]], [P[2, end]], "X", ms = 14, mfc = "#c62828", mec = "k",
                mew = 1.4, zorder = 10, label = "breakdown")
    end

    ax.set_title("(a) Phase portrait" * tag * "  →  " * which_crit(st.solution),
                 fontsize = 11, fontweight = "bold")
    ax.set_xlabel("\$x\$"); ax.set_ylabel("\$y\$")
    ax.legend(loc = "lower left", fontsize = 8, framealpha = 0.93)
    ax.set_xlim(-0.95, 1.30); ax.set_ylim(-0.85, 0.90)
    ax.set_aspect("equal"); ax.grid(alpha = 0.14)

    # ---------- (b) function values ----------
    ax = ax_b
    ax.plot(0:length(fv)-1, fv, "-o", color = "#0d47a1", ms = 4, lw = 1.6,
            mfc = "white", mew = 1.2)
    ax.axhline(f(SADDLE), color = "crimson", ls = "--", lw = 1.3,
               label = "\$f(x^*) = -27/256\$")
    ax.axhline(-0.125, color = "#2e7d32", ls = ":", lw = 1.3,
               label = "\$f\$ at minima \$= -1/8\$")
    ax.axhline(0.0, color = "0.55", ls = ":", lw = 1.0, label = "\$f(0,0) = 0\$")
    ax.set_title("(b) Function values \$f(x_k)\$", fontsize = 11, fontweight = "bold")
    ax.set_xlabel("iteration \$k\$"); ax.set_ylabel("\$f(x_k)\$")
    K > 200 && ax.set_xscale("symlog")
    ax.legend(fontsize = 8.5); ax.grid(alpha = 0.25)

    # ---------- (c) radius and step length ----------
    ax = ax_c
    ax.semilogy(0:length(Δv)-1, max.(Δv, 1e-18), "-o", color = "#1565c0", ms = 4,
                lw = 1.6, mfc = "white", mew = 1.2, label = "\$\\Delta_k\$")
    if K > 0
        ax.semilogy(0:K-1, max.(sv, 1e-18), "--s", color = "#ef6c00", ms = 3.5, lw = 1.3,
                    alpha = 0.9, label = "\$\\|s_k\\|\$")
    end
    if K <= 200                                # active shading only when legible
        for i in 1:K
            av[i] && ax.axvspan(i-1.5, i-0.5, color = "#1565c0", alpha = 0.07, zorder = 0)
        end
    end
    ax.set_title("(c) Trust-region radius", fontsize = 11, fontweight = "bold")
    ax.set_xlabel("iteration \$k\$"); ax.set_ylabel("\$\\Delta_k\$, \$\\|s_k\\|\$")
    K > 200 && ax.set_xscale("symlog")
    ax.legend(fontsize = 9); ax.grid(alpha = 0.25, which = "both")

    # ---------- (d) ρ_k ----------
    ax = ax_d
    KR = collect(0:K-1)
    fin = findall(isfinite, ρv)                # ρ = -Inf when pred ≤ 0; drop, don't plot
    KR, R = KR[fin], ρv[fin]
    # three thresholds now, not two: η accepts, η1 and η2 scale.
    cols = [v >= η2 ? "#2e7d32" : (v >= η1 ? "#f9a825" :
            (v >= η  ? "#7e57c2" : "#c62828")) for v in R]
    if length(KR) <= 200
        ax.bar(KR, R, color = cols, edgecolor = "k", lw = 0.6, width = 0.66, zorder = 3)
    else
        ax.scatter(KR, R, c = cols, s = 6, edgecolors = "none", zorder = 3)
        ax.set_xscale("symlog")
    end
    ax.axhline(η2, color = "#2e7d32", ls = "--", lw = 1.2,
               label = "\$\\eta_2 = $(round(η2, digits=3))\$")
    ax.axhline(η1, color = "#f9a825", ls = "--", lw = 1.2,
               label = "\$\\eta_1 = $(round(η1, digits=3))\$")
    η < η1 && ax.axhline(η, color = "#7e57c2", ls = "-.", lw = 1.2,
                         label = "\$\\eta = $(round(η, digits=3))\$ (accept)")
    ax.axhline(1.0, color = "0.35", ls = "--", lw = 1.2, label = "\$\\rho = 1\$")
    ax.set_title("(d) Ratio \$\\rho_k\$" * (η < η1 ? "  (purple: accepted, radius contracts)" : ""),
                 fontsize = 11, fontweight = "bold")
    ax.set_xlabel("iteration \$k\$"); ax.set_ylabel("\$\\rho_k\$")
    ax.legend(fontsize = 8.5); ax.grid(alpha = 0.25, axis = "y")

    PyPlot.tight_layout()
    savepath === nothing || PyPlot.savefig(savepath, dpi = 165, bbox_inches = "tight")
    return fig
end

"""
    diagnose(; rule, model, subsolver, x0, kmax, tol, tol_H, savepath)

Run one configuration through `tr_solve` and produce the four-panel diagnostic, labelled
automatically from the types.
"""
function diagnose(; rule::RadiusRule, model::ModelHessian = ExactHessian(),
                    subsolver::SubproblemSolver = SteihaugCG(),
                    x0 = X0_DEFAULT, kmax = 4000, tol = 1e-8, tol_H = -1.0,
                    savepath = nothing)
    st, xs = solve_path(rule = rule, model = model, subsolver = subsolver,
                        x0 = x0, kmax = kmax, tol = tol, tol_H = tol_H)
    @printf("%s + %s + %s : %s in %d iters -> %s\n",
            nameof(typeof(model)), nameof(typeof(rule)), nameof(typeof(subsolver)),
            st.status, st.iter, which_crit(st.solution))
    return plot_run(st, xs;
                    rule_name  = string(nameof(typeof(rule))),
                    model_name = string(nameof(typeof(model))),
                    solver_name = string(nameof(typeof(subsolver))),
                    savepath = savepath)
end