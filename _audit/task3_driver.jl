# =============================================================================
# _audit/task3_driver.jl
#
# The computation behind mu_cap_gradient_descent_v1.ipynb, in one file so that it
# can be run and checked before the notebook is assembled from it. The notebook
# cells are cut from this file at the `#%%` markers, so the two cannot disagree.
#
#   julia --project=notebooks _audit/task3_driver.jl
# =============================================================================

#%% setup
using TrustRegionRadius
using ADNLPModels, NLPModels
using LinearAlgebra, Printf, Statistics
using Plots, LaTeXStrings

gr()
default(fontfamily = "Computer Modern", framestyle = :box, grid = true,
        gridalpha = 0.25, legendfontsize = 7, guidefontsize = 9,
        titlefontsize = 10, tickfontsize = 8)

include(joinpath(@__DIR__, "..", "notebooks", "Saddle", "saddle_problem.jl"))

# Black plus one accent, separated by dash pattern and marker as well, so that
# nothing is carried by colour alone.
const BLACK  = RGB(0.0, 0.0, 0.0)
const ACCENT = RGB(0.85, 0.33, 0.10)
const GREY   = RGB(0.62, 0.62, 0.62)

const REPDIR = joinpath(@__DIR__, "..", "report", "task3-mu-cap")
const OUTDIR = joinpath(REPDIR, "figs")
mkpath(OUTDIR)

#%% grid
const MU_SWEEP   = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 128.0]
const ZETA_SWEEP = [0.1, 0.5, 1.0, 2.0, 5.0, 20.0]

# Keyword arguments everywhere. RGrad and RGradCapped were renumbered, γ2 now
# being the mild contraction and γ3 the expansion, so a positional call can pass
# silently under the wrong reading.
mk_cap(μm) = RGradCapped(γ1 = G1, γ2 = G2, γ3 = G3, μ = min(MU0, μm), μ_max = μm)
mk_dfo(ζ)  = RDFO(γ1 = G1, γ2 = G2, γ3 = G3, ζ = ζ)

const RULE_SETTINGS = vcat([(kind = :cap, param = μm, mk = () -> mk_cap(μm))
                            for μm in MU_SWEEP],
                           [(kind = :dfo, param = ζ, mk = () -> mk_dfo(ζ))
                            for ζ in ZETA_SWEEP])
const SUBSOLVERS = [("SteihaugCG", () -> SteihaugCG()),
                    ("ExactMS",    () -> ExactMS())]
const MODELS = [("ExactHessian", () -> ExactHessian()),
                ("SR1",          () -> SR1Model(mem = 10)),
                ("LBFGS",        () -> LBFGSModel(mem = 5))]

#%% runner
"""
    run_counted(; rule, model, subsolver, x0) -> (stats, path, evals)

One `tr_solve` call, keeping the model so that the evaluation counters survive.

`run_cfg` of `saddle_problem.jl` builds its `ADNLPModel` inline and returns only
the statistics, so `neval_obj` and its companions are unreachable afterwards.
Section 4.4 asks for a work measure that survives a change of budget, and an
iteration count is not one. The iteration is still entirely `tr_solve`'s, and the
path comes from the solver's own callback exactly as `solve_path` does.
"""
function run_counted(; rule, model = ExactHessian(), subsolver = SteihaugCG(),
                       x0 = X0_DEFAULT)
    nlp = make_nlp(x0)
    xs  = [collect(float.(x0))]
    st = tr_solve(nlp; rule = rule, model = model, subsolver = subsolver,
                  params = TRParams(η = ETA1, η1 = ETA1, η2 = ETA2, Δ0 = DELTA0,
                                    tol = TOL, tol_H = -1.0,
                                    max_iterations = KMAX),
                  trace = true, hessian_norm = true,
                  callback = (_n, _s, stats) -> push!(xs, copy(stats.solution)))
    return st, xs, (f = neval_obj(nlp), g = neval_grad(nlp), h = neval_hprod(nlp))
end

#%% classify
"""
    step_classes(ss) -> (truncated, inside, krylov)

The three classes of Section 4.1, as fractions of the iterations that happened.

1. truncated at the boundary, `cg_iters == 1` and the step active: the Hessian
   touched neither the direction nor the length;
2. stopped inside after one CG iteration: the direction is `-g_k` and the Hessian
   sets the length through the exact line search on the model;
3. two or more CG iterations.

`ExactMS` reports `cg_iters == 0` throughout, so the classes do not apply and the
caller reads the angle instead.
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

"Least squares slope of log10(y) on log10(x), with intercept, R² and count."
function loglog_fit(x, y; ymin = 0.0)
    idx = [i for i in eachindex(y) if isfinite(x[i]) && isfinite(y[i]) &&
                                      x[i] > 0 && y[i] > ymin]
    length(idx) < 5 && return (slope = NaN, inter = NaN, r2 = NaN, n = length(idx))
    X, Y = log10.(x[idx]), log10.(y[idx])
    X̄, Ȳ = mean(X), mean(Y)
    sxx = sum(abs2, X .- X̄)
    sxx == 0 && return (slope = NaN, inter = NaN, r2 = NaN, n = length(idx))
    b = sum((X .- X̄) .* (Y .- Ȳ)) / sxx
    a = Ȳ - b * X̄
    ss = sum(abs2, Y .- Ȳ)
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

#%% sweep
function run_grid()
    rows = NamedTuple[]; paths = Dict{String, Any}()
    for (snm, mksub) in SUBSOLVERS, (mnm, mkmod) in MODELS, rs in RULE_SETTINGS
        st, xs, ev = run_counted(rule = rs.mk(), model = mkmod(), subsolver = mksub())
        ss = st.solver_specific
        cosc = ss[:cos_cauchy_trajectory]; Δ = ss[:delta_trajectory]
        cg = ss[:cg_iters_trajectory];     g = ss[:grad_trajectory]
        c1, c2, c3 = step_classes(ss)
        nc = nearest_crit(st.solution)
        cauchy = isempty(cosc) ? NaN :
                 count(x -> isfinite(x) && abs(x - 1) <= 1e-12, cosc) / length(cosc)
        direct = (isempty(cg) || all(==(0), cg)) ? NaN :
                 count(==(1), cg) / length(cg)
        key = "$(snm)|$(mnm)|$(rs.kind)|$(rs.param)"
        paths[key] = (xs = xs, Δ = Δ, cos = cosc, cg = cg,
                      ac = ss[:active_trajectory], g = g)
        push!(rows, (subsolver = snm, model = mnm, kind = rs.kind,
                     param = rs.param, key = key, status = st.status, iter = st.iter,
                     f_evals = ev.f, g_evals = ev.g, h_evals = ev.h,
                     gnorm = Float64(st.dual_feas), obj = Float64(st.objective),
                     limit = nc.name, dist = nc.dist,
                     delta_end = Δ[end], delta_min = minimum(Δ),
                     cauchy = cauchy, direct = direct,
                     trunc_frac = c1, inside_frac = c2, krylov_frac = c3,
                     cos_min = isempty(cosc) ? NaN : minimum(cosc),
                     cos_frac = isempty(cosc) ? NaN :
                                count(x -> isfinite(x) && x > 1 - 1e-12, cosc) / length(cosc),
                     active = active_fraction(st; tail = 0.1)))
    end
    return rows, paths
end

@info "running the grid: $(length(SUBSOLVERS)*length(MODELS)*length(RULE_SETTINGS)) configurations"
const T0 = time()
ROWS, PATHS = run_grid()
@info "grid done" seconds = round(time() - T0, digits = 1) rows = length(ROWS)
sel(; kw...) = [r for r in ROWS if all(getfield(r, k) == v for (k, v) in kw)]

#%% table1
for (ttl, kd, pnm) in (("TABLE 1. RGradCapped, the cap sweep.", :cap, "mu_max"),
                       ("TABLE 2. RDFO, the zeta sweep.",       :dfo, "zeta"))
    println("\n", "="^102); println(ttl, " Three step classes, Steihaug CG."); println("="^102)
    @printf("%-14s %8s %11s %7s %8s %8s %8s %9s %9s %10s\n", "model", pnm,
            "status", "iter", "trunc", "inside", "krylov", "cauchy", "cg1", "cos_min")
    println("-"^102)
    for (mnm, _) in MODELS, r in sel(subsolver = "SteihaugCG", model = mnm, kind = kd)
        @printf("%-14s %8.2f %11s %7d %8.3f %8.3f %8.3f %9.4f %9.4f %10.6f\n",
                mnm, r.param, string(r.status), r.iter, r.trunc_frac,
                r.inside_frac, r.krylov_frac, r.cauchy, r.direct, r.cos_min)
    end
end

#%% reconcile
println("\n", "="^108)
println("TABLE 3. The two Cauchy statistics reconciled. Exact Hessian only, since")
println("H_k is known in closed form there. A disagreeing iteration is one the")
println("cosine test counts and the Krylov count does not.")
println("="^108)
@printf("%-6s %8s %9s %9s %8s %7s %11s %11s %13s %13s\n", "rule", "param",
        "cauchy", "cg1", "gap", "n_dis", "sin@dis", "sin@agree", "z1/D@dis", "z1/D@agree")
println("-"^108)
RECON = NamedTuple[]
for rs in RULE_SETTINGS
    p = PATHS["SteihaugCG|ExactHessian|$(rs.kind)|$(rs.param)"]
    n = length(p.cg); n == 0 && continue
    dis = [i for i in 1:n if abs(p.cos[i] - 1) <= 1e-12 && p.cg[i] >= 2]
    agr = [i for i in 1:n if !(abs(p.cos[i] - 1) <= 1e-12) && p.cg[i] >= 2]
    med(v) = isempty(v) ? NaN : median(v)
    sd = med([eig_deviation(p.xs[i]) for i in dis])
    sa = med([eig_deviation(p.xs[i]) for i in agr])
    zd = med([z1_ratio(p.xs[i], p.Δ[i]) for i in dis])
    za = med([z1_ratio(p.xs[i], p.Δ[i]) for i in agr])
    r = only(sel(subsolver = "SteihaugCG", model = "ExactHessian",
                 kind = rs.kind, param = rs.param))
    push!(RECON, (kind = rs.kind, param = rs.param, cauchy = r.cauchy,
                  direct = r.direct, gap = r.cauchy - r.direct,
                  ndis = length(dis), sin_dis = sd, sin_agree = sa,
                  z1_dis = zd, z1_agree = za))
    @printf("%-6s %8.2f %9.4f %9.4f %8.4f %7d %11.3e %11.3e %13.9f %13.9f\n",
            string(rs.kind), rs.param, r.cauchy, r.direct, r.cauchy - r.direct,
            length(dis), sd, sa, zd, za)
end

#%% angle-fit
println("\n", "="^96)
println("TABLE 4. ExactMS: the exponent of 1 - cos(s,-g), fitted three ways.")
println("="^96)
# (a) against Delta, as the brief asks, one fit per configuration
@printf("(a) 1 - cos against Delta, per configuration\n")
@printf("%-14s %-6s %8s %10s %8s %8s\n", "model", "rule", "param", "slope", "R2", "n")
println("-"^60)
FITS_DELTA = NamedTuple[]
for (mnm, _) in MODELS, rs in RULE_SETTINGS
    p = PATHS["ExactMS|$(mnm)|$(rs.kind)|$(rs.param)"]
    n = length(p.cos); n == 0 && continue
    ft = loglog_fit(p.Δ[1:n], [1 - c for c in p.cos]; ymin = 0.0)
    push!(FITS_DELTA, (model = mnm, kind = rs.kind, param = rs.param,
                       slope = ft.slope, r2 = ft.r2, n = ft.n))
    ft.n >= 30 && @printf("%-14s %-6s %8.2f %10.4f %8.4f %8d\n",
                          mnm, string(rs.kind), rs.param, ft.slope, ft.r2, ft.n)
end
let ok = [f for f in FITS_DELTA if isfinite(f.slope) && f.n >= 30]
    @printf("\n  %d fits with n >= 30: slope median %.4f, range [%.4f, %.4f], R2 range [%.3f, %.3f]\n",
            length(ok), median([f.slope for f in ok]),
            minimum(f.slope for f in ok), maximum(f.slope for f in ok),
            minimum(f.r2 for f in ok), maximum(f.r2 for f in ok))
end

# (b) against the parameter the expansion is actually in
# s* = -(H + sigma I)^{-1} g with sigma set by ||s*|| = Delta. Expanding for large
# sigma in the eigenbasis, and using sigma ~ ||g||/Delta,
#     1 - cos  ~  (1/2) Var_w(lambda) mu^2,    mu = Delta/||g||,
# with w_i the weights of g on the eigenvectors. The expansion parameter is mu,
# not Delta. RGradCapped pins mu at mu_bar within a run, so the variation has to
# come from pooling the runs.
MU = Float64[]; YY = Float64[]; DD = Float64[]; RAW = Float64[]
for rs in RULE_SETTINGS
    rs.kind === :cap || continue
    p = PATHS["ExactMS|ExactHessian|cap|$(rs.param)"]
    for i in eachindex(p.cos)
        x = p.xs[i]; ng = norm(grad(x)); ng == 0 && continue
        v = spectrum_variance(x); y = 1 - p.cos[i]
        (isfinite(y) && y > 0 && isfinite(v) && v > 1e-12) || continue
        push!(MU, p.Δ[i] / ng); push!(YY, y / v); push!(DD, p.Δ[i]); push!(RAW, y)
    end
end
const FIT_POOLED = loglog_fit(DD, RAW)
const FIT_MU     = loglog_fit(MU, YY)
const SMALL      = [i for i in eachindex(MU) if MU[i] <= 0.5]
const FIT_MU_SM  = loglog_fit(MU[SMALL], YY[SMALL])
@printf("\n(b) pooled over the cap sweep, exact Hessian, %d points, mu in [%.3g, %.3g]\n",
        length(MU), minimum(MU), maximum(MU))
@printf("    1 - cos            against Delta : slope %+.4f  R2 %.4f  n %d\n",
        FIT_POOLED.slope, FIT_POOLED.r2, FIT_POOLED.n)
@printf("    (1 - cos)/Var_w    against mu    : slope %+.4f  R2 %.4f  n %d  intercept %+.4f\n",
        FIT_MU.slope, FIT_MU.r2, FIT_MU.n, FIT_MU.inter)
@printf("    same, mu <= 0.5                  : slope %+.4f  R2 %.4f  n %d  intercept %+.4f\n",
        FIT_MU_SM.slope, FIT_MU_SM.r2, FIT_MU_SM.n, FIT_MU_SM.inter)
@printf("    predicted                        : slope %+.4f              intercept %+.4f\n",
        2.0, log10(0.5))

# (c) SteihaugCG on the truncated branch: the same quantity is identically zero
println("\n(c) SteihaugCG, exact Hessian: iterations with 1 - cos exactly zero")
@printf("%-6s %8s %10s %10s %10s\n", "rule", "param", "n", "n(zero)", "share")
for rs in RULE_SETTINGS
    p = PATHS["SteihaugCG|ExactHessian|$(rs.kind)|$(rs.param)"]
    n = length(p.cos); n == 0 && continue
    z = count(c -> c == 1.0, p.cos)
    @printf("%-6s %8.2f %10d %10d %10.4f\n", string(rs.kind), rs.param, n, z, z / n)
end

#%% limits
println("\n", "="^118)
println("TABLE 5. Limit points, as distance rather than as a category. Nothing is dropped.")
println("="^118)
@printf("%-11s %-14s %-6s %8s %11s %7s %11s %11s %8s %8s %8s\n", "subsolver",
        "model", "rule", "param", "status", "iter", "dist", "gnorm", "f_ev", "g_ev", "active")
println("-"^118)
for r in ROWS
    @printf("%-11s %-14s %-6s %8.2f %11s %7d %11.3e %11.3e %8d %8d %8.3f\n",
            r.subsolver, r.model, string(r.kind), r.param, string(r.status),
            r.iter, r.dist, r.gnorm, r.f_evals, r.g_evals, r.active)
end

#%% figures
# --- Figure 1: Delta_k against k for a few caps -------------------------------
let sel_mu = [0.05, 0.5, 2.0, 128.0]
    plt = plot(xlabel = L"k", ylabel = L"\Delta_k", yscale = :log10,
               legend = :bottomleft, size = (620, 400))
    for (j, μm) in enumerate(sel_mu)
        p = PATHS["SteihaugCG|ExactHessian|cap|$μm"]
        plot!(plt, 0:length(p.Δ)-1, max.(p.Δ, 1e-18); label = L"\bar{\mu} = %$μm",
              color = isodd(j) ? BLACK : ACCENT,
              linestyle = (:solid, :dash, :dot, :dashdot)[j], lw = 1.6)
    end
    savefig(plt, joinpath(OUTDIR, "fig1_radius_vs_k.pdf"))
end

# --- Figure 2: the three classes stacked, one panel per model ----------------
# Stacked by hand, drawing the cumulative heights from the back forwards.
# `groupedbar` lives in StatsPlots, which this environment does not carry.
let panels = []
    for (mnm, _) in MODELS
        rs = sel(subsolver = "SteihaugCG", model = mnm, kind = :cap)
        xs = 1:length(rs)
        t = [r.trunc_frac for r in rs]; i2 = [r.inside_frac for r in rs]
        k = [r.krylov_frac for r in rs]
        p = bar(xs, t .+ i2 .+ k; label = "2+ CG", color = GREY, lw = 0,
                title = mnm, xticks = (xs, string.([r.param for r in rs])),
                xrotation = 60, ylabel = "fraction of iterations",
                xlabel = L"\bar{\mu}", ylims = (0, 1.02),
                legend = (mnm == "ExactHessian" ? :bottomleft : false))
        bar!(p, xs, t .+ i2; label = "inside, 1 CG", color = ACCENT, lw = 0)
        bar!(p, xs, t;       label = "truncated",    color = BLACK,  lw = 0)
        push!(panels, p)
    end
    savefig(plot(panels...; layout = (1, 3), size = (1250, 420)),
            joinpath(OUTDIR, "fig2_step_classes.pdf"))
end

# --- Figure 3: the angle, as asked and as diagnosed --------------------------
let μm = 0.05
    pe = PATHS["ExactMS|ExactHessian|cap|$μm"]
    ps = PATHS["SteihaugCG|ExactHessian|cap|$μm"]
    ne, ns = length(pe.cos), length(ps.cos)
    ye, de = [1 - c for c in pe.cos], pe.Δ[1:ne]
    ys, ds = [1 - c for c in ps.cos], ps.Δ[1:ns]

    p1 = scatter(de[ye .> 0], ye[ye .> 0]; xscale = :log10, yscale = :log10,
                 label = "ExactMS", color = ACCENT, ms = 2.2, msw = 0,
                 xlabel = L"\Delta_k", ylabel = L"1 - \cos(s_k, -g_k)",
                 legend = :bottomright, title = "as asked, against " * L"\Delta_k")
    let z = [i for i in 1:ns if ys[i] == 0]
        isempty(z) || scatter!(p1, ds[z], fill(1e-17, length(z));
                               label = "SteihaugCG, truncated (exactly 0)",
                               color = BLACK, ms = 2.6, msw = 0)
    end
    let nz = [i for i in 1:ns if ys[i] > 0]
        isempty(nz) || scatter!(p1, ds[nz], ys[nz]; label = "SteihaugCG, not truncated",
                                color = BLACK, ms = 2.6, msw = 0, marker = :xcross)
    end
    annotate!(p1, exp10(mean(log10.(de[ye .> 0]))), maximum(ye) * 0.5,
              text(@sprintf("pooled slope %.2f, R² %.2f", FIT_POOLED.slope,
                            FIT_POOLED.r2), 7))

    p2 = scatter(MU, YY; xscale = :log10, yscale = :log10, label = "ExactMS, all caps",
                 color = ACCENT, ms = 2.2, msw = 0, xlabel = L"\mu_k = \Delta_k/\|g_k\|",
                 ylabel = L"(1-\cos)/\mathrm{Var}_w(\lambda)", legend = :bottomright,
                 title = "against the expansion parameter")
    let xs = exp10.(range(log10(minimum(MU)), log10(maximum(MU)); length = 40))
        plot!(p2, xs, 10 .^ (FIT_MU_SM.inter .+ FIT_MU_SM.slope .* log10.(xs));
              label = @sprintf("fit, slope %.3f (R² %.4f)", FIT_MU_SM.slope, FIT_MU_SM.r2),
              color = BLACK, linestyle = :dash, lw = 2)
        plot!(p2, xs, 0.5 .* xs .^ 2; label = L"\frac{1}{2}\mu^2 \ \mathrm{(predicted)}",
              color = BLACK, linestyle = :dot, lw = 2)
    end
    savefig(plot(p1, p2; layout = (1, 2), size = (1180, 450)),
            joinpath(OUTDIR, "fig3_angle_vs_delta.pdf"))
end

# --- Figure 4: iterate paths over the contours -------------------------------
let
    xr = range(-0.9, 1.3; length = 260); yr = range(-0.9, 0.9; length = 260)
    Z = [f([a, b]) for b in yr, a in xr]
    plt = contour(xr, yr, Z; levels = 26, color = :grays, colorbar = false,
                  xlabel = L"p_1", ylabel = L"p_2", size = (660, 470), legend = :topleft)
    for (j, μm) in enumerate([0.05, 128.0])
        p = PATHS["SteihaugCG|ExactHessian|cap|$μm"]
        P = reduce(hcat, p.xs)
        plot!(plt, P[1, :], P[2, :]; label = L"\bar{\mu} = %$μm",
              color = isodd(j) ? BLACK : ACCENT,
              linestyle = isodd(j) ? :solid : :dash, lw = 1.5)
    end
    for (nm, q) in CRITPTS
        scatter!(plt, [q[1]], [q[2]]; label = "", color = ACCENT, ms = 5, marker = :star5)
        annotate!(plt, q[1], q[2] + 0.07, text(nm, 7))
    end
    scatter!(plt, [X0_DEFAULT[1]], [X0_DEFAULT[2]]; label = L"x_0", color = BLACK,
             ms = 5, marker = :square)
    savefig(plt, joinpath(OUTDIR, "fig4_paths.pdf"))
end
@info "figures written" OUTDIR

#%% latex
# Every number the report states is emitted here, so none is transcribed by hand.
esc_(s) = replace(String(s), "_" => "\\_")
open(joinpath(REPDIR, "numbers.tex"), "w") do io
    cmd(k, v) = println(io, "\\newcommand{\\", k, "}{", v, "}")
    cmd("NConfigs", length(ROWS))
    cmd("NCaps", length(MU_SWEEP)); cmd("NZetas", length(ZETA_SWEEP))
    cmd("FitDeltaSlope", @sprintf("%.2f", FIT_POOLED.slope))
    cmd("FitDeltaRsq",   @sprintf("%.2f", FIT_POOLED.r2))
    cmd("FitMuSlope",    @sprintf("%.3f", FIT_MU_SM.slope))
    cmd("FitMuRsq",      @sprintf("%.4f", FIT_MU_SM.r2))
    cmd("FitMuInter",    @sprintf("%.3f", FIT_MU_SM.inter))
    cmd("FitMuN",        FIT_MU_SM.n)
    cmd("FitMuPred",     @sprintf("%.3f", log10(0.5)))
    for r in RECON
        r.kind === :cap || continue
        (r.param in (4.0, 8.0)) || continue
        t = r.param == 4.0 ? "Four" : "Eight"
        cmd("Gap$t",    @sprintf("%.4f", r.gap))
        cmd("NDis$t",   r.ndis)
        cmd("SinDis$t",   @sprintf("%.3f", r.sin_dis))
        cmd("SinAgree$t", @sprintf("%.3f", r.sin_agree))
        cmd("ZDis$t",     @sprintf("%.9f", r.z1_dis))
        cmd("ZAgree$t",   @sprintf("%.4f", r.z1_agree))
    end
    let r = only(sel(subsolver = "SteihaugCG", model = "ExactHessian", kind = :cap, param = 4.0))
        cmd("CauchyFour", @sprintf("%.4f", r.cauchy)); cmd("CgOneFour", @sprintf("%.4f", r.direct))
    end
    let r = only(sel(subsolver = "SteihaugCG", model = "ExactHessian", kind = :cap, param = 8.0))
        cmd("CauchyEight", @sprintf("%.4f", r.cauchy)); cmd("CgOneEight", @sprintf("%.4f", r.direct))
    end
end

open(joinpath(REPDIR, "tab_classes.tex"), "w") do io
    println(io, "\\begin{tabular}{@{}lrrrrrrr@{}}"); println(io, "\\toprule")
    println(io, "Model & \$\\bar{\\mu}\$ & status & iter & trunc.\\ & inside & 2+ CG & \$\\min\\cos\$ \\\\ ")
    println(io, "\\midrule")
    for (j, (mnm, _)) in enumerate(MODELS)
        for (i, r) in enumerate(sel(subsolver = "SteihaugCG", model = mnm, kind = :cap))
            @printf(io, "%s & %.2f & \\texttt{%s} & %d & %.3f & %.3f & %.3f & %.6f \\\\ \n",
                    i == 1 ? esc_(mnm) : "", r.param, esc_(string(r.status)), r.iter,
                    r.trunc_frac, r.inside_frac, r.krylov_frac, r.cos_min)
        end
        j < length(MODELS) && println(io, "\\midrule")
    end
    println(io, "\\bottomrule"); println(io, "\\end{tabular}")
end

open(joinpath(REPDIR, "tab_reconcile.tex"), "w") do io
    println(io, "\\begin{tabular}{@{}lrrrrrrrr@{}}"); println(io, "\\toprule")
    println(io, "Rule & param & cauchy & \\texttt{cg=1} & gap & \$n\$ & " *
                "\$\\sin\\angle\$ dis. & \$\\sin\\angle\$ agr. & \$\\|z_1\\|/\\Delta\$ dis. \\\\ ")
    println(io, "\\midrule")
    for r in RECON
        r.ndis == 0 && continue
        @printf(io, "%s & %.2f & %.4f & %.4f & %.4f & %d & %.3f & %.3f & %.9f \\\\ \n",
                r.kind === :cap ? "cap" : "DFO", r.param,
                r.cauchy, r.direct, r.gap, r.ndis,
                r.sin_dis, r.sin_agree, r.z1_dis)
    end
    println(io, "\\bottomrule"); println(io, "\\end{tabular}")
end

open(joinpath(REPDIR, "tab_limits.tex"), "w") do io
    println(io, "\\begin{tabular}{@{}llrrrrrr@{}}"); println(io, "\\toprule")
    println(io, "Subsolver & Model & \$\\bar{\\mu}\$ & status & iter & " *
                "dist & \$\\|g\\|\$ & \$\\nabla f\$ evals \\\\ ")
    println(io, "\\midrule")
    for r in ROWS
        r.kind === :cap || continue
        r.model == "ExactHessian" || continue
        @printf(io, "%s & %s & %.2f & \\texttt{%s} & %d & %.3e & %.3e & %d \\\\ \n",
                esc_(r.subsolver), esc_(r.model), r.param, esc_(string(r.status)),
                r.iter, r.dist, r.gnorm, r.g_evals)
    end
    println(io, "\\bottomrule"); println(io, "\\end{tabular}")
end

@info "latex fragments written" REPDIR
println("\nDRIVER OK")
