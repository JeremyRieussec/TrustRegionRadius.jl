# =============================================================================
# benchmark/experiments/exp17_second_order_profiles.jl
#
# EXPERIMENT 17 -- performance profiles for the second-order rules on CUTEst.
#
# Three profiles, in this order of importance.
#
#   P1  among the second-order rules, on the second-order task.
#   P2  second-order against first-order, scored on the FIRST-order task.
#   P3  second-order against first-order, scored on the SECOND-order task.
#
# P2 and P3 together are the comparison of the two families. P2 alone says the
# second-order arm is expensive, P3 alone says it is reliable, and either one on
# its own has misdescribed the experiment.
#
# ---------------------------------------------------------------------------
# THE SCORING RULE
#
# Every run is scored afterwards, by ONE criterion applied by the same code to
# every run, never by the solver's own status. A first-order rule stops when the
# gradient is small; a second-order rule keeps going until it also has curvature.
# Profiling those two on their own stopping tests compares the cost of an easy
# task with the cost of a hard one, and the first-order arm wins by construction.
#
# The criterion is `second_order_status(‖g‖, λ_min, ε_g, ε_H)` of the package,
# read at the point the run returned.
#
# ---------------------------------------------------------------------------
# WHY THIS FILE HAS ITS OWN RUN LOOP
#
# Two things `run_experiment` cannot express, both of them decisive here.
#
#   1. `RunRecord` carries `h_evals = neval_hprod`, and no `neval_hess`. With
#      `ExactHessian` and n ≤ 200 the curvature estimate goes through
#      `dense_hessian`, hence through `NLPModels.hess`, so on the small problems
#      the entire cost of the second-order machinery lands in a counter the
#      shared record does not keep. A profile on Hessian-vector products alone
#      would show the second-order arm as free, which is the opposite of true.
#   2. The subproblem solver of the second-order arm depends on the DIMENSION of
#      the problem (`ExactMS` at n ≤ 200, `EigenPoint(SteihaugCG())` above), and
#      a config factory takes no arguments and never sees n.
#
# Nothing under `src/`, no existing experiment file, `config.jl` or `harness.jl`
# was modified. The archive layer, the profile functions and the criterion are
# the package's own.
#
#   julia --project=benchmark -e 'include("benchmark/initialisation.jl"); second_order_profiles()'
#
# Environment overrides, all optional:
#
#   TRR_SO_MAXVAR    upper bound on n           (default MAX_VAR of config.jl)
#   TRR_SO_LIMIT     first N problems only      (default all; for a pilot)
#   TRR_SO_MAXTIME   seconds per run            (default SOLVER_PARAMS.max_time)
#   TRR_SO_RESUME    an existing archive to continue into
# =============================================================================

# -----------------------------------------------------------------------------
# The two tolerances, fixed once
# -----------------------------------------------------------------------------
# ε_g is SOLVER_PARAMS.tol, so the first-order task is the one every other
# experiment in this suite solves. ε_H is SECOND_ORDER_PARAMS.tol_H.
const SOP_EPS_G = SOLVER_PARAMS.tol            # 1e-5
const SOP_EPS_H = SECOND_ORDER_PARAMS.tol_H    # 1e-6

# The dense/exact boundary. `ExactMS` refuses above its `nmax`, `ExactHessian`'s
# curvature estimate falls back to Lanczos above `curv_nmax`, and a Ritz value is
# an upper bound on λ_min, so the second-order test becomes optimistic there.
# P3 is restricted to n ≤ SOP_NMAX for exactly that reason.
const SOP_NMAX = 200

sop_int_env(k, default) = haskey(ENV, k) ? parse(Int, ENV[k]) : default
sop_flt_env(k, default) = haskey(ENV, k) ? parse(Float64, ENV[k]) : default

const SOP_MAXVAR  = sop_int_env("TRR_SO_MAXVAR", MAX_VAR)
const SOP_LIMIT   = haskey(ENV, "TRR_SO_LIMIT") ? parse(Int, ENV["TRR_SO_LIMIT"]) : nothing
const SOP_MAXTIME = sop_flt_env("TRR_SO_MAXTIME", SOLVER_PARAMS.max_time)

# -----------------------------------------------------------------------------
# The roster
# -----------------------------------------------------------------------------
# Six rules, three of each kind. The third column is `is_criticality_anchored`,
# asserted against the package below rather than trusted.
#
# Rdfo, Rgrad and Rgrtr READ a criticality measure, so ω_k = τ_k changes the rule
# itself. Rdelta, Rstep and Rrtr do not read one, so for them only the stopping
# test and the subproblem solver change between the arms. Both kinds belong in
# P1 and the caption must say which is which, or the figure claims a distinction
# it does not have.
#
# Constants are config.jl's, shared: γ1, γ2, γ3 = 0.25, 0.50, 2.0, ζ = 100, μ = 1,
# Δmin = 0 on every rule. Per-rule tuning would measure tuning effort.
const SOP_RULES = [
    ("Rdelta", () -> RDelta(   γ1 = G1, γ2 = G2, γ3 = G3, Δmin = 0.0),           false),
    ("Rstep",  () -> RStep(    γ1 = G1, γ2 = G2, γ3 = G3, Δmin = 0.0),           false),
    ("Rrtr",   () -> RRTR(     γ1 = G1, γ2 = G2, γ3 = G3, Δmin = 0.0),           false),
    ("Rdfo",   () -> RDFO(     γ1 = G1, γ2 = G2, γ3 = G3, ζ = 100.0, Δmin = 0.0), true),
    ("Rgrad",  () -> RGrad(    γ1 = G1, γ2 = G2, γ3 = G3, μ = 1.0, Δmin = 0.0),   true),
    ("Rgrtr",  () -> RRTRGrad( γ1 = G1, γ2 = G2, γ3 = G3, μ = 1.0, Δmin = 0.0),   true),
]

const SOP_ARMS  = (:fo, :so)
const SOP_PASSES = (:g, :h)          # :g scores the first-order task, :h the second

"""
    sop_rule(factory, anchored, arm) -> RadiusRule

The rule as the arm uses it. In the second-order arm a criticality-anchored rule
is wrapped in `SecondOrder`, which overrides `criticality` and forwards
`update_radius!` unchanged, so ω_k = τ_k and nothing else moves. A rule that is
not criticality-anchored never reads the measure, so it is passed through: the
second-order arm changes only its stopping test and its subproblem solver.
"""
sop_rule(factory, anchored::Bool, arm::Symbol) =
    (arm === :so && anchored) ? SecondOrder(factory()) : factory()

"""
    sop_subsolver(arm, n) -> SubproblemSolver

The first-order arm takes `DEFAULT_SUBSOLVER` of config.jl.

The second-order arm takes a solver that delivers the eigenpoint decrease of
`as: Eigenpoint decrease`: `ExactMS` at n ≤ 200, which solves the subproblem
exactly including the hard case, and `EigenPoint(SteihaugCG())` above, which
returns whichever of the truncated-CG step and ±Δv_min decreases the model more.
Plain `SteihaugCG` delivers neither: it stops at the first negative-curvature
direction the recurrence happens to reach and its guaranteed decrease carries no
`|λ_min|Δ²` term.

The switch is on the problem, not on the column, so every column sees the same
subproblem solver on a given problem and the per-problem normalisation of a
performance profile is unaffected.
"""
sop_subsolver(arm::Symbol, n::Int) =
    arm === :fo ? DEFAULT_SUBSOLVER() :
    n <= SOP_NMAX ? ExactMS(nmax = SOP_NMAX) :
                    EigenPoint(SteihaugCG(max_iters = 1_000))

"""
    sop_params(arm, pass) -> TRParams

`SOLVER_PARAMS` throughout, with two fields moved.

`tol = ε_g` in every configuration. `tol_H = ε_H` in the second-order arm and
`-1` in the first-order arm, in BOTH passes: `tol_H` controls the curvature
estimate as well as the stopping test, so switching it off in pass `:g` would
also switch off the cost that pass exists to measure.

Pass `:g` stops at the first iterate with `‖g_k‖ ≤ ε_g` through the callback of
`sop_solve`, not through `tol_H`.
"""
sop_params(arm::Symbol, ::Symbol) =
    TRParams(η  = SOLVER_PARAMS.η, η1 = SOLVER_PARAMS.η1, η2 = SOLVER_PARAMS.η2,
             Δ0 = SOLVER_PARAMS.Δ0, Δmin = SOLVER_PARAMS.Δmin,
             Δmax = SOLVER_PARAMS.Δmax,
             max_iterations = SOLVER_PARAMS.max_iterations,
             tol = SOP_EPS_G,
             tol_H = arm === :so ? SOP_EPS_H : -1.0,
             max_time = SOP_MAXTIME,
             curv_nmax = SOP_NMAX)

sop_label(rname, arm) = string(rname, arm === :so ? " (2nd)" : " (1st)")
sop_key(rname, arm, pass) = string(rname, "_", arm, "_", pass)

# -----------------------------------------------------------------------------
# One run
# -----------------------------------------------------------------------------

"""
    sop_solve(nlp, rname, factory, anchored, arm, pass) -> NamedTuple

One `tr_solve`, with the four evaluation counters read off the model afterwards.

`NLPModels.reset!` first, so the counters describe this run alone.

Pass `:g` halts at the first iterate whose gradient norm is at most ε_g, through
a callback rather than through the solver's own test. That keeps each arm's own
configuration intact while making the FIRST-ORDER point the stopping point, which
is what P2 asks for and what a post hoc score cannot supply: the trace carries no
per-iteration evaluation counts, so the cost a run had spent when it first met a
criterion is not recoverable from an archived run.

The callback runs at the end of every iteration and never at iteration 0, so a
starting point that is already first-order costs one iteration more than it
should. `k0_first_order` records that case; it is expected to be empty on CUTEst.
"""
function sop_solve(nlp, rname, factory, anchored::Bool, arm::Symbol, pass::Symbol)
    n = nlp.meta.nvar
    NLPModels.reset!(nlp)
    rule = sop_rule(factory, anchored, arm)
    sub  = sop_subsolver(arm, n)
    p    = sop_params(arm, pass)

    cb = pass === :g ?
        ((nlp_, solver_, stats_) ->
             (stats_.dual_feas <= SOP_EPS_G && TrustRegionRadius.SolverCore.set_status!(stats_, :user))) :
        ((args...) -> nothing)

    t0 = time()
    st = tr_solve(nlp; rule = rule, model = ExactHessian(), subsolver = sub,
                  params = p, trace = true, callback = cb)
    wall = time() - t0

    ss = st.solver_specific
    gtraj = get(ss, :grad_trajectory,       Float64[])
    ltraj = get(ss, :lambda_min_trajectory, Float64[])

    return (problem = nlp.meta.name, n = n, rule = rname, arm = arm, pass = pass,
            status  = st.status,
            iters   = st.iter,
            f_evals = neval_obj(nlp),
            g_evals = neval_grad(nlp),
            h_prods = neval_hprod(nlp),
            h_evals = neval_hess(nlp),
            gnorm   = Float64(st.dual_feas),
            lam     = isempty(ltraj) ? NaN : ltraj[end],
            obj     = Float64(st.objective),
            wall    = wall,
            k0_first_order = !isempty(gtraj) && gtraj[1] <= SOP_EPS_G)
end

sop_row_keys() = (:problem, :n, :rule, :arm, :pass, :status, :iters, :f_evals,
                  :g_evals, :h_prods, :h_evals, :gnorm, :lam, :obj, :wall,
                  :k0_first_order)

function sop_save(arch, row)
    save_data(arch, data_filename(row.problem, sop_key(row.rule, row.arm, row.pass));
              problem = row.problem, n = row.n, rule = row.rule,
              arm = string(row.arm), pass = string(row.pass),
              status = row.status, iters = row.iters,
              f_evals = row.f_evals, g_evals = row.g_evals,
              h_prods = row.h_prods, h_evals = row.h_evals,
              gnorm = row.gnorm, lam = row.lam, obj = row.obj, wall = row.wall,
              k0_first_order = row.k0_first_order)
end

function sop_load(arch, pname, rname, arm, pass)
    d = load_data(arch, pname, sop_key(rname, arm, pass))
    return (problem = d["problem"], n = d["n"], rule = d["rule"],
            arm = Symbol(d["arm"]), pass = Symbol(d["pass"]),
            status = d["status"], iters = d["iters"],
            f_evals = d["f_evals"], g_evals = d["g_evals"],
            h_prods = d["h_prods"], h_evals = d["h_evals"],
            gnorm = d["gnorm"], lam = d["lam"], obj = d["obj"], wall = d["wall"],
            k0_first_order = d["k0_first_order"])
end

sop_failed_row(pname, n, rname, arm, pass) =
    (problem = pname, n = n, rule = rname, arm = arm, pass = pass,
     status = :exception, iters = 0, f_evals = 0, g_evals = 0, h_prods = 0,
     h_evals = 0, gnorm = NaN, lam = NaN, obj = NaN, wall = 0.0,
     k0_first_order = false)

# -----------------------------------------------------------------------------
# The problem set
# -----------------------------------------------------------------------------

"""
    sop_problems() -> Vector{Tuple{String, Function}}

The unconstrained CUTEst set, with no fallback.

`default_problems()` returns `analytic_problems()` when the CUTEst query comes
back empty, so a run in which CUTEst failed to load produces a table that reads
as a CUTEst benchmark and is not one. This raises instead.
"""
function sop_problems()
    HAS_CUTEST || error("exp17: CUTEst is unavailable. This benchmark is a CUTEst " *
                        "benchmark and there is no analytic fallback: a table " *
                        "produced on `analytic_problems()` and labelled CUTEst " *
                        "would be a false statement about the problem set.")
    ps = cutest_problems(min_var = MIN_VAR, max_var = SOP_MAXVAR,
                         max_con = MAX_CON, limit = SOP_LIMIT)
    isempty(ps) && error("exp17: the CUTEst query returned no problems at " *
                         "min_var = $MIN_VAR, max_var = $SOP_MAXVAR, " *
                         "max_con = $MAX_CON.")
    return ps
end

# -----------------------------------------------------------------------------
# The campaign
# -----------------------------------------------------------------------------

"""
    sop_campaign(arch, problems) -> (rows, dims)

Every problem, every rule, both arms, both passes. The model is opened once per
problem, which on CUTEst decodes and compiles a SIF file and dominates the cost
of a resumed run.

Resumable: a `(problem, rule, arm, pass)` already in `data/` is read back rather
than recomputed, so an interrupted campaign continues where it stopped.
"""
function sop_campaign(arch, problems)
    rows = NamedTuple[]
    dims = Dict{String, Int}()
    ncfg = length(SOP_RULES) * length(SOP_ARMS) * length(SOP_PASSES)
    nreused = 0

    for (ip, (pname, mk)) in enumerate(problems)
        cached = Dict{String, Any}()
        for (rname, _, _) in SOP_RULES, arm in SOP_ARMS, pass in SOP_PASSES
            has_data(arch, pname, sop_key(rname, arm, pass)) || continue
            try
                cached[sop_key(rname, arm, pass)] = sop_load(arch, pname, rname, arm, pass)
            catch
                # A JLD2 truncated by an interruption is unreadable; treat it as
                # absent and recompute.
            end
        end

        if length(cached) == ncfg
            for r in values(cached)
                push!(rows, r); dims[pname] = r.n
            end
            nreused += ncfg
            @printf("[%3d/%3d] %-12s  all %d cached\n", ip, length(problems), pname, ncfg)
            continue
        end

        nlp = try
            mk()
        catch err
            @warn "exp17: could not open problem" problem = pname err
            for (rname, _, _) in SOP_RULES, arm in SOP_ARMS, pass in SOP_PASSES
                push!(rows, sop_failed_row(pname, 0, rname, arm, pass))
            end
            dims[pname] = 0
            continue
        end
        n = nlp.meta.nvar
        dims[pname] = n

        t_pb = time()
        for (rname, factory, anchored) in SOP_RULES, arm in SOP_ARMS, pass in SOP_PASSES
            key = sop_key(rname, arm, pass)
            if haskey(cached, key)
                push!(rows, cached[key]); nreused += 1; continue
            end
            row = try
                sop_solve(nlp, rname, factory, anchored, arm, pass)
            catch err
                err isa InterruptException && rethrow()
                @warn "exp17: run failed" problem = pname rule = rname arm = arm pass = pass err
                sop_failed_row(pname, n, rname, arm, pass)
            end
            # The record must carry the CUTEst name, not the model's own.
            row = merge(row, (problem = pname,))
            push!(rows, row)
            try
                sop_save(arch, row)
            catch err
                err isa InterruptException && rethrow()
                @warn "exp17: could not archive run" problem = pname config = key err
            end
        end
        finalize(nlp)
        @printf("[%3d/%3d] %-12s  n = %5d   %6.1f s\n",
                ip, length(problems), pname, n, time() - t_pb)
        flush(stdout)
    end

    nreused > 0 && @printf("\n%d run(s) read from the archive, %d computed.\n",
                           nreused, length(rows) - nreused)
    return rows, dims
end

# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------
# One criterion, the package's own, applied by this code to every run.

"""
    sop_ok_first(r) -> Bool

The first-order criterion at the returned point: `‖g‖ ≤ ε_g`.
"""
sop_ok_first(r) = isfinite(r.gnorm) && r.gnorm <= SOP_EPS_G

"""
    sop_ok_second(r) -> Bool

The second-order criterion at the returned point:
`‖g‖ ≤ ε_g` and `λ_min ≥ −ε_H`, through `second_order_status` of the package.

`λ_min` is the model's, and the model is `ExactHessian`, so at n ≤ SOP_NMAX it is
the exact leftmost eigenvalue of `∇²f(x)` from a dense symmetric
eigendecomposition. This is the reason P3 is restricted to that subset: above it
the estimate is a Ritz value, which over-states λ_min and would score the runs by
the machinery the profile exists to compare.
"""
sop_ok_second(r) =
    isfinite(r.gnorm) && isfinite(r.lam) &&
    second_order_status(r.gnorm, r.lam, SOP_EPS_G, SOP_EPS_H) === :second_order

const SOP_METRICS = (
    (:h_prods, "Hessian-vector products"),
    (:h_evals, "Hessian evaluations"),
    (:g_evals, "gradient evaluations"),
    (:f_evals, "objective evaluations"),
    (:iters,   "iterations"),
)

"""
    sop_matrix(rows, pnames, cols, pass, metric, ok) -> Matrix{Float64}

The cost matrix a profile consumes: `Inf` wherever the criterion `ok` was not met
at the returned point, whatever the solver's own status said.
"""
function sop_matrix(rows, pnames, cols, pass::Symbol, metric::Symbol, ok::Function)
    idx = Dict((r.problem, r.rule, r.arm, r.pass) => r for r in rows)
    M = fill(Inf, length(pnames), length(cols))
    for (i, p) in enumerate(pnames), (j, (rname, arm)) in enumerate(cols)
        r = get(idx, (p, rname, arm, pass), nothing)
        r === nothing && continue
        ok(r) || continue
        v = Float64(getproperty(r, metric))
        M[i, j] = v > 0 ? v : 1.0     # a zero cost would divide the profile by 0
    end
    return M
end

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

function sop_profile_figure(arch, fname, M, labels, xlab, title)
    τ, prof = performance_profile(M)
    plt = plot(τ, prof; xscale = :log10, label = reshape(labels, 1, :),
               xlabel = "τ (performance ratio, $xlab)", ylabel = "π(τ)",
               title = title, titlefontsize = 9,
               legend = :bottomright, lw = 2, ylims = (0, 1.02))
    savefig_archived(arch, fname, plt)
    open(joinpath(arch.tables, replace(fname, ".pdf" => ".tex")), "w") do io
        profile_to_pgfplots(io, τ, prof, labels)
    end
    return nothing
end

function sop_data_figure(arch, fname, N, dims, labels, title)
    κ, dp = data_profile(N, dims)
    plt = plot(κ, dp; label = reshape(labels, 1, :),
               xlabel = "budget (simplex gradients, (n+1) evaluations)",
               ylabel = "fraction solved", title = title, titlefontsize = 9,
               legend = :bottomright, lw = 2, ylims = (0, 1.02))
    savefig_archived(arch, fname, plt)
    open(joinpath(arch.tables, replace(fname, ".pdf" => ".tex")), "w") do io
        profile_to_pgfplots(io, κ, dp, labels; logx = false)
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------

const SOP_STATUSES = (:first_order, :second_order, :user, :max_iter, :max_time,
                      :stalled, :exception)

"""
    sop_status_table(rows, pnames, cols, pass, ok, okname) -> String

Status composition per configuration, with no filtering on success and no mean
over survivors. The last two columns are the criterion this experiment scores on,
which is not the solver's status and can disagree with it in both directions.
"""
function sop_status_table(rows, pnames, cols, pass::Symbol, ok::Function, okname)
    idx = Dict((r.problem, r.rule, r.arm, r.pass) => r for r in rows)
    io = IOBuffer()
    @printf(io, "%-16s", "configuration")
    for s in SOP_STATUSES; @printf(io, "%11s", string(s)); end
    @printf(io, "%11s%11s\n", okname, "missing")
    println(io, "-"^(16 + 11 * (length(SOP_STATUSES) + 2)))
    for (rname, arm) in cols
        counts = Dict(s => 0 for s in SOP_STATUSES)
        nok = 0; nmiss = 0; nother = 0
        for p in pnames
            r = get(idx, (p, rname, arm, pass), nothing)
            if r === nothing
                nmiss += 1; continue
            end
            haskey(counts, r.status) ? (counts[r.status] += 1) : (nother += 1)
            ok(r) && (nok += 1)
        end
        @printf(io, "%-16s", sop_label(rname, arm))
        for s in SOP_STATUSES; @printf(io, "%11d", counts[s]); end
        @printf(io, "%11d%11d\n", nok, nmiss)
        nother > 0 && @printf(io, "%-16s%11d  (statuses outside the listed set)\n",
                              "", nother)
    end
    println(io)
    @printf(io, "%d problems, pass :%s, criterion %s, eps_g = %g, eps_H = %g\n",
            length(pnames), pass, okname, SOP_EPS_G, SOP_EPS_H)
    return String(take!(io))
end

"""
    sop_cost_table(rows, pnames, cols, pass, ok) -> String

Total cost over ALL problems and, separately, over the problems on which the
criterion was met. The first is the honest aggregate; the second is reported
beside it and labelled, so that a reader can see how much of the difference
between two columns is a difference in what they solved.
"""
function sop_cost_table(rows, pnames, cols, pass::Symbol, ok::Function)
    idx = Dict((r.problem, r.rule, r.arm, r.pass) => r for r in rows)
    io = IOBuffer()
    @printf(io, "%-16s %8s %12s %12s %12s %12s %10s\n", "configuration", "met",
            "f evals", "g evals", "H-prods", "H evals", "wall (s)")
    println(io, "-"^96)
    for (rname, arm) in cols
        nok = 0
        tf = tg = thp = the = 0; tw = 0.0
        for p in pnames
            r = get(idx, (p, rname, arm, pass), nothing)
            r === nothing && continue
            tf += r.f_evals; tg += r.g_evals; thp += r.h_prods; the += r.h_evals
            tw += r.wall
            ok(r) && (nok += 1)
        end
        @printf(io, "%-16s %5d/%-3d %12d %12d %12d %12d %10.1f\n",
                sop_label(rname, arm), nok, length(pnames), tf, tg, thp, the, tw)
    end
    println(io)
    println(io, "Totals are over every problem, met or not. No filtering on ",
                "success and no mean over survivors.")
    return String(take!(io))
end

# -----------------------------------------------------------------------------
# The three checks of section 3 of the brief
# -----------------------------------------------------------------------------

"""
    sop_checks(problems) -> String

Three things established before the campaign is read, each one a check that can
fail rather than a claim.

  1. `is_criticality_anchored` on every rule in the roster, against the package.
  2. `τ` as the package computes it, against `eqn:tau definition` of Part II,
     `τ_k = max{‖g_k‖, −λ_k^-}` with `λ_k^-` the leftmost eigenvalue of the MODEL
     Hessian. Checked on a grid of `(‖g‖, λ)` including `λ > 0`, `λ = 0`, `λ < 0`.
  3. `ExactHessian`'s dense branch against `NLPModels.hess`, which is what makes
     `λ_min(B_k) = λ_min(∇²f(x_k))` and so makes P3's criterion a statement about
     the true Hessian rather than about a model.

Plus the `LBFGSModel` degeneracy of section 4 of the brief: `B ≻ 0` by
construction, so `τ ≡ ‖g‖` and the second-order variant is the first-order one.
Verified on iterate sequences, not asserted.
"""
function sop_checks(problems)
    io = IOBuffer()
    println(io, "CHECK 1 -- which rules read a criticality measure")
    println(io, "-"^70)
    for (rname, factory, anchored) in SOP_RULES
        r = factory()
        got = is_criticality_anchored(r)
        @printf(io, "  %-10s is_criticality_anchored = %-5s  regime = %-14s %s\n",
                rname, got, asymptotic_regime(r), got == anchored ? "ok" : "MISMATCH")
        got == anchored || error("exp17: roster disagrees with the package on $rname")
    end
    println(io)

    println(io, "CHECK 2 -- tau against eqn:tau definition of Part II")
    println(io, "-"^70)
    worst = 0.0
    for gn in (0.0, 1e-8, 1.0, 7.5), λ in (-3.0, -1e-9, 0.0, 1e-9, 2.0)
        got  = tau_criticality(gn, λ)
        want = max(gn, -λ)                 # max{‖g‖, −λ^-}, λ^- the leftmost eigenvalue
        worst = max(worst, abs(got - want))
        @printf(io, "  |g| = %-8g  lambda = %-10g  package %-12g  Part II %-12g\n",
                gn, λ, got, want)
    end
    @printf(io, "  worst absolute difference: %.3e\n", worst)
    worst == 0 || error("exp17: tau_criticality does not match eqn:tau definition")
    println(io)
    println(io, "  Part I defines no second-order criticality measure. Its")
    println(io, "  criticality-anchored section carries chi_k, a generic measure with")
    println(io, "  chi_k = |g_k| as its example, and it states that second-order")
    println(io, "  stationarity is outside its scope. The word eigenvalue does not")
    println(io, "  occur in it. So there is no second definition to choose between.")
    println(io)

    println(io, "CHECK 3 -- ExactHessian's dense branch is the true Hessian")
    println(io, "-"^70)
    nchecked = 0; worst3 = 0.0
    for (pname, mk) in problems
        nchecked >= 5 && break
        nlp = try mk() catch; continue end
        n = nlp.meta.nvar
        if n > SOP_NMAX
            finalize(nlp); continue
        end
        x = copy(nlp.meta.x0)
        B = Symmetric(Matrix(dense_hessian(ExactHessian(), nlp, x)))
        H = Symmetric(Matrix(TrustRegionRadius._full_hessian(hess(nlp, x))))
        d1 = maximum(abs, Matrix(B) .- Matrix(H))
        lam_pkg  = lambda_min_estimate(ExactHessian(), nlp, x; nmax = SOP_NMAX)
        lam_dir  = minimum(eigvals(H))
        d2 = abs(lam_pkg - lam_dir)
        worst3 = max(worst3, max(d1, d2))
        @printf(io, "  %-12s n = %4d  max|B - H| = %.3e   |lambda_pkg - lambda_dense| = %.3e\n",
                pname, n, d1, d2)
        nchecked += 1
        finalize(nlp)
    end
    @printf(io, "  worst over %d problems: %.3e\n", nchecked, worst3)
    println(io)

    println(io, "CHECK 4 -- LBFGSModel: the second-order variant IS the first-order one")
    println(io, "-"^70)
    println(io, "  B is positive definite by construction, so lambda_min > 0, so")
    println(io, "  tau = |g| at every iteration. LBFGSModel therefore gets no line")
    println(io, "  of its own in P1. The two arms are compared on iterate sequences.")
    ndiff = 0; nsame = 0
    for (pname, mk) in problems
        nsame + ndiff >= 6 && break
        nlp = try mk() catch; continue end
        if nlp.meta.nvar > SOP_NMAX
            finalize(nlp); continue
        end
        for (rname, factory, anchored) in SOP_RULES
            anchored || continue
            p = TRParams(η = SOLVER_PARAMS.η, η1 = SOLVER_PARAMS.η1,
                         η2 = SOLVER_PARAMS.η2, Δ0 = SOLVER_PARAMS.Δ0,
                         tol = SOP_EPS_G, max_iterations = 300, max_time = 30.0)
            seqs = map((false, true)) do second
                NLPModels.reset!(nlp)
                rl = second ? SecondOrder(factory()) : factory()
                st = tr_solve(nlp; rule = rl, model = LBFGSModel(),
                              subsolver = DEFAULT_SUBSOLVER(), params = p, trace = true)
                (st.iter, copy(st.solver_specific[:grad_trajectory]),
                 copy(st.solver_specific[:delta_trajectory]))
            end
            a, b = seqs
            same = a[1] == b[1] && a[2] == b[2] && a[3] == b[3]
            same ? (nsame += 1) : (ndiff += 1)
            @printf(io, "  %-12s %-8s iters %4d vs %4d   trajectories %s\n",
                    pname, rname, a[1], b[1], same ? "identical" : "DIFFER")
            if !same
                m = min(length(a[2]), length(b[2]))
                k = findfirst(i -> a[2][i] != b[2][i], 1:m)
                @printf(io, "  %-12s %-8s first difference at k = %s\n",
                        "", "", k === nothing ? "none in the common prefix" : string(k - 1))
            end
        end
        finalize(nlp)
    end
    @printf(io, "  identical: %d   differing: %d\n", nsame, ndiff)
    ndiff > 0 && println(io, "  A difference is a FINDING, reported and not smoothed over.")
    return String(take!(io))
end

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

function second_order_profiles()
    arch = haskey(ENV, "TRR_SO_RESUME") ?
        reopen_archive(ENV["TRR_SO_RESUME"]) :
        ExperimentArchive(tag = "second_order_profiles")

    problems = sop_problems()
    pnames   = [p[1] for p in problems]

    configs = Tuple{String, Function}[]
    for (rname, factory, anchored) in SOP_RULES, arm in SOP_ARMS
        push!(configs, (sop_label(rname, arm),
                        () -> (rule = sop_rule(factory, anchored, arm),
                               model = ExactHessian(),
                               subsolver = sop_subsolver(arm, SOP_NMAX))))
    end

    save_config(arch; configs = configs, params = SOLVER_PARAMS,
                problem_selection = Dict("min_var" => MIN_VAR,
                                         "max_var" => SOP_MAXVAR,
                                         "max_con" => MAX_CON),
                extra = Dict("experiment"      => "exp17_second_order_profiles",
                             "eps_g"           => SOP_EPS_G,
                             "eps_H"           => SOP_EPS_H,
                             "exact_nmax"      => SOP_NMAX,
                             "max_time"        => SOP_MAXTIME,
                             "cutest_problems" => length(problems),
                             "scoring"         => "post hoc, second_order_status " *
                                                  "at the returned point; never the " *
                                                  "solver's own status"))

    @info "Experiment 17: $(length(problems)) CUTEst problems × $(length(SOP_RULES)) rules × 2 arms × 2 passes"

    checks = sop_checks(problems)
    save_table(arch, "exp17_checks.txt", checks)
    println(checks)

    rows, dims = sop_campaign(arch, problems)
    sop_analyse(arch, rows, dims, pnames)
    return rows
end

"""
    sop_analyse(arch, rows, dims, pnames) -> nothing

Every table and every figure, from rows already in hand. Split out of
`second_order_profiles` so that `second_order_profiles_report` can redraw a
campaign from its archive without recomputing a single run.
"""
function sop_analyse(arch, rows, dims, pnames)
    cols_so  = [(rname, :so) for (rname, _, _) in SOP_RULES]
    cols_all = vcat([[(rname, :fo), (rname, :so)] for (rname, _, _) in SOP_RULES]...)
    # --- the realised problem list, names and dimensions -----------------------
    io = IOBuffer()
    @printf(io, "%-14s %8s\n", "problem", "n")
    println(io, "-"^24)
    nsmall = 0
    for p in pnames
        n = get(dims, p, 0)
        @printf(io, "%-14s %8d\n", p, n)
        0 < n <= SOP_NMAX && (nsmall += 1)
    end
    @printf(io, "\n%d problems, %d with n <= %d (the P3 subset), %d above.\n",
            length(pnames), nsmall, SOP_NMAX, length(pnames) - nsmall)
    save_table(arch, "exp17_problems.txt", String(take!(io)))

    small = [p for p in pnames if 0 < get(dims, p, 0) <= SOP_NMAX]
    dimvec(ps) = Float64[get(dims, p, 1) for p in ps]

    lab(cols) = [sop_label(r, a) for (r, a) in cols]

    # ---- P1: among the second-order rules, on the second-order task -----------
    for (metric, xlab) in SOP_METRICS
        M = sop_matrix(rows, pnames, cols_so, :h, metric, sop_ok_second)
        sop_profile_figure(arch, "exp17_P1_$(metric).pdf", M, lab(cols_so), xlab,
                           "P1: second-order arm, second-order criterion")
    end
    N1 = sop_matrix(rows, pnames, cols_so, :h, :f_evals, sop_ok_second)
    sop_data_figure(arch, "exp17_P1_data.pdf", N1, dimvec(pnames), lab(cols_so),
                    "P1: data profile, objective evaluations")
    save_table(arch, "exp17_P1_status.txt",
               sop_status_table(rows, pnames, cols_so, :h, sop_ok_second, "2nd-order"))
    save_table(arch, "exp17_P1_cost.txt",
               sop_cost_table(rows, pnames, cols_so, :h, sop_ok_second))

    # ---- P2: both arms, first-order criterion ---------------------------------
    for (metric, xlab) in SOP_METRICS
        M = sop_matrix(rows, pnames, cols_all, :g, metric, sop_ok_first)
        sop_profile_figure(arch, "exp17_P2_$(metric).pdf", M, lab(cols_all), xlab,
                           "P2: both arms, first-order criterion")
    end
    N2 = sop_matrix(rows, pnames, cols_all, :g, :f_evals, sop_ok_first)
    sop_data_figure(arch, "exp17_P2_data.pdf", N2, dimvec(pnames), lab(cols_all),
                    "P2: data profile, objective evaluations")
    save_table(arch, "exp17_P2_status.txt",
               sop_status_table(rows, pnames, cols_all, :g, sop_ok_first, "1st-order"))
    save_table(arch, "exp17_P2_cost.txt",
               sop_cost_table(rows, pnames, cols_all, :g, sop_ok_first))

    # ---- P3: both arms, second-order criterion, n <= SOP_NMAX only ------------
    for (metric, xlab) in SOP_METRICS
        M = sop_matrix(rows, small, cols_all, :h, metric, sop_ok_second)
        sop_profile_figure(arch, "exp17_P3_$(metric).pdf", M, lab(cols_all), xlab,
                           "P3: both arms, second-order criterion, n <= $SOP_NMAX")
    end
    N3 = sop_matrix(rows, small, cols_all, :h, :f_evals, sop_ok_second)
    sop_data_figure(arch, "exp17_P3_data.pdf", N3, dimvec(small), lab(cols_all),
                    "P3: data profile, objective evaluations")
    save_table(arch, "exp17_P3_status.txt",
               sop_status_table(rows, small, cols_all, :h, sop_ok_second, "2nd-order"))
    save_table(arch, "exp17_P3_cost.txt",
               sop_cost_table(rows, small, cols_all, :h, sop_ok_second))

    # ---- hazards --------------------------------------------------------------
    io = IOBuffer()
    nk0 = count(r -> r.k0_first_order, rows)
    @printf(io, "H1  runs whose STARTING point already met |g| <= eps_g: %d\n", nk0)
    println(io, "    The pass-:g callback fires at the end of an iteration and never at")
    println(io, "    iteration 0, so such a run costs one iteration more than it should.")
    println(io, "    Expected to be 0 on CUTEst.")
    nlanczos = count(p -> get(dims, p, 0) > SOP_NMAX, pnames)
    @printf(io, "\nH2  problems above n = %d, where lambda_min is a Ritz value: %d\n",
            SOP_NMAX, nlanczos)
    println(io, "    A Ritz value over-states lambda_min, so the second-order test is")
    println(io, "    optimistic there. P3 excludes them; P1 and P2 include them and")
    println(io, "    P1's certificate on those problems is Lanczos-limited.")
    ndis = count(rows) do r
        r.pass === :h && (r.status === :second_order) != sop_ok_second(r)
    end
    @printf(io, "\nH3  pass :h runs where the solver's status and the post hoc criterion\n")
    @printf(io, "    disagree: %d\n", ndis)
    println(io, "    They can disagree in both directions. This is why the profiles")
    println(io, "    are scored post hoc and never on the status.")
    nexc = count(r -> r.status === :exception, rows)
    @printf(io, "\nH4  runs that raised: %d\n", nexc)
    save_table(arch, "exp17_hazards.txt", String(take!(io)))

    finalize_archive(arch; notes = """
        Performance profiles for the second-order radius rules on CUTEst.

        Every run is scored afterwards, by one criterion applied by the same code
        to every run, and never by the solver's own status. The two families stop
        on different tests, so profiling them on their own statuses would compare
        the cost of an easy task with the cost of a hard one.

        eps_g = $(SOP_EPS_G), eps_H = $(SOP_EPS_H).

        P1 is the second-order arm on the second-order criterion. P2 and P3 are
        both arms, on the first-order and the second-order criterion in turn.
        Read P2 and P3 together: P2 is the overhead, P3 is the payoff, and either
        one alone has misdescribed the comparison.

        Two passes per configuration, because the trace carries no per-iteration
        evaluation counts and the cost a run had spent when it first met a
        criterion is therefore not recoverable from an archived run. Pass :g stops
        at the first iterate with |g| <= eps_g, pass :h at the first that also has
        lambda_min >= -eps_H. Both passes keep their arm's own tol_H, so the
        second-order arm pays for its curvature estimate in both.

        Profiles are given on Hessian-vector products and on Hessian evaluations
        separately. With ExactHessian at n <= $(SOP_NMAX) the curvature estimate
        goes through NLPModels.hess, so on the small problems the whole cost of
        the second-order machinery is in the Hessian-evaluation count and none of
        it in the Hessian-vector count.

        These are average-case results on one problem set. Nothing here says
        anything about the sharpness of a worst-case complexity bound.
        """)
    return nothing
end


"""
    second_order_profiles_report(dir; require_complete = true) -> Vector

Redraw every table and figure of a campaign from its archive, recomputing
nothing.

`require_complete` keeps only the problems for which all
`length(SOP_RULES) * 4` runs are present, so a partial campaign yields a
coherent profile on a stated subset rather than a profile in which a missing
run is indistinguishable from a failure. The realised subset is written to
`exp17_problems.txt` and printed.
"""
function second_order_profiles_report(dir::AbstractString; require_complete::Bool = true)
    arch = reopen_archive(dir)
    ncfg = length(SOP_RULES) * length(SOP_ARMS) * length(SOP_PASSES)
    rows = NamedTuple[]
    for f in sort(readdir(arch.data))
        endswith(f, ".jld2") || continue
        d = try
            JLD2.load(joinpath(arch.data, f))
        catch
            @warn "unreadable, skipped" file = f
            continue
        end
        push!(rows, (problem = d["problem"], n = d["n"], rule = d["rule"],
                     arm = Symbol(d["arm"]), pass = Symbol(d["pass"]),
                     status = d["status"], iters = d["iters"],
                     f_evals = d["f_evals"], g_evals = d["g_evals"],
                     h_prods = d["h_prods"], h_evals = d["h_evals"],
                     gnorm = d["gnorm"], lam = d["lam"], obj = d["obj"],
                     wall = d["wall"], k0_first_order = d["k0_first_order"]))
    end
    isempty(rows) && error("exp17: no readable runs in $dir")

    counts = Dict{String, Int}()
    dims   = Dict{String, Int}()
    for r in rows
        counts[r.problem] = get(counts, r.problem, 0) + 1
        dims[r.problem] = r.n
    end
    pnames = sort(collect(keys(counts)))
    require_complete && (pnames = [p for p in pnames if counts[p] == ncfg])
    isempty(pnames) && error("exp17: no problem has all $ncfg runs archived")
    keep = Set(pnames)
    rows = [r for r in rows if r.problem in keep]

    @printf("exp17 report: %d complete problems, %d runs, from %s\n",
            length(pnames), length(rows), dir)
    sop_analyse(arch, rows, dims, pnames)
    return rows
end
