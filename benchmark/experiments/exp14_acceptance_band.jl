# =============================================================================
# benchmark/experiments/exp14_acceptance_band.jl
#
# EXPERIMENT 14 -- what it costs to accept a step the radius does not trust.
#
# `TRParams` carries 0 ≤ η ≤ η1 ≤ η2 < 1. η decides acceptance and is read by
# the solver alone. η1 and η2 decide scaling and are read by the rule alone.
# Setting η < η1 opens a band, ρ_k ∈ [η, η1), on which the step is accepted and
# the radius still contracts. The band is empty by construction at η = η1.
#
# This file sweeps η from 0 to η1 at fixed η1 and measures three things, in this
# order, because the third is meaningless without the first:
#
#   1. OCCUPANCY.  How often the band is entered at all. The effect cannot
#      exceed the exposure. Counted as `accepted[i] && ρ[i] < η1` on accepted
#      iterations, NOT as a radius decrease: RDelta contracts by γ2 on every
#      mildly successful iteration, so a shrinking radius does not isolate the
#      band. The count is zero whenever η = η1, which is the check that the
#      instrument works.
#
#   2. WORK, paired by problem. A rejected step leaves the iterate in place and
#      pays one objective evaluation. An accepted step also pays a gradient:
#      `grad!` sits inside `if st.accepted` in src/Trust-region/common.jl.
#      Lowering η therefore converts cheap iterations into expensive ones, and an
#      iteration count will show a gain the evaluation count may not. Function,
#      gradient and Hessian-vector counts are reported separately for that
#      reason, and iterations are secondary.
#
#   3. THE η = 0 COLUMN, twice. At η = 0 every step with a nonnegative ratio is
#      accepted. Near the solution the achieved reduction falls to the rounding
#      level of f and ρ stops carrying information, so that column is the one
#      most exposed to an arithmetic artefact. Accepted iterations whose
#      predicted reduction sits below eps * max(1, |f_k|) are counted, and the
#      column is reported with and without them.
#
# WHY THIS FILE DOES NOT CALL `run_experiment`
#
# `run_experiment` takes ONE `TRParams` for every configuration, and this sweep
# varies η per configuration. Calling it once per η value would re-decode every
# SIF file five times per stratum. It also builds a `RunRecord`, which does not
# carry `:accepted_trajectory`, and the occupancy count of point 1 cannot be
# formed without it. So the loop below opens each CUTEst model once and calls
# `tr_solve` -- the package entry point, not a local solver -- against every
# configuration in turn, and reads `solver_specific` directly.
#
#   julia --project=benchmark benchmark/experiments/exp14_acceptance_band.jl
#
# Resumable: one JLD2 per problem, so an interrupted campaign continues with
#
#   TRR_RESUME=benchmark/results/exp_..._acceptance_band \
#     julia --project=benchmark benchmark/experiments/exp14_acceptance_band.jl
#
# `TRR_AB_LIMIT=n` runs the first n problems only. That is a pilot, not the
# experiment: the realised count is written into the archive so a pilot archive
# cannot be mistaken for a full one.
# =============================================================================

# -----------------------------------------------------------------------------
# The sweep
# -----------------------------------------------------------------------------

"η as a fraction of η1. The last is the classical coupling and the baseline."
const AB_ETA_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]

# Two strata. Changing η1 changes the rule's own behaviour, so the second is a
# separate axis and is never pooled with the first. A band of width 0.1 in ρ may
# be too narrow to separate anything, which is the whole reason for the second.
const AB_STRATA = [(:narrow, 0.1), (:wide, 0.4)]

# The eight of `tab: roster axes` marked "Tested", which is exactly config.jl's
# RULES. Held at one set of constants, as config.jl already does.
const AB_RULES = RULES

# Surveyed in `tab: mechanism summary` of Part III and marked untested there.
# Reported in a separate table, never as part of the primary eight.
const AB_RULES_UNTESTED = [
    ("RDeltaStep",          () -> RDeltaStep(γ1 = G1, γ2 = G2, γ3 = G3, Δmin = 0.0)),
    ("RAdaptiveGradCapped", () -> RAdaptiveGradCapped(μ = 1.0, μ_max = 128.0,
                                                      λ1 = 5.0, λ2 = 5.0, Δmin = 0.0)),
    ("RRTRGrad",            () -> RRTRGrad(γ1 = G1, γ2 = G2, γ3 = G3, μ = 1.0,
                                           Δmin = 0.0)),
]

"Tie band on the per-problem log ratio, below which a pair counts as a tie."
const AB_TIE = 0.05

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

"""
    ab_params(η1, frac) -> TRParams

`SOLVER_PARAMS` with η1 set to the stratum and η set to `frac * η1`. Everything
else is held fixed, so the swept threshold is the only difference between two
configurations of the same rule.
"""
ab_params(η1::Float64, frac::Float64) =
    TRParams(η              = frac * η1,
             η1             = η1,
             η2             = SOLVER_PARAMS.η2,
             Δ0             = SOLVER_PARAMS.Δ0,
             Δmin           = SOLVER_PARAMS.Δmin,
             Δmax           = SOLVER_PARAMS.Δmax,
             max_iterations = SOLVER_PARAMS.max_iterations,
             tol            = SOLVER_PARAMS.tol,
             tol_H          = SOLVER_PARAMS.tol_H,
             max_time       = SOLVER_PARAMS.max_time)

"Every (stratum, rule, η) triple, as (label, stratum, rule name, factory, frac, η1)."
function ab_configs(rules = AB_RULES)
    out = NamedTuple[]
    for (sname, η1) in AB_STRATA, (rname, rf) in rules, frac in AB_ETA_FRACTIONS
        push!(out, (label  = @sprintf("%s|%s|%.2f", sname, rname, frac),
                    strat  = sname, rule = rname, factory = rf,
                    frac   = frac,  η1 = η1,      η = frac * η1))
    end
    return out
end

# -----------------------------------------------------------------------------
# One run
# -----------------------------------------------------------------------------

"""
    ab_run(nlp, cfg) -> NamedTuple

Solve once and reduce the trace to what the analysis needs.

The two counts that matter are formed here rather than stored as trajectories,
because 185 problems times 80 configurations of full traces is not a file anyone
opens twice.

`band` is the occupancy of `eqn: band` -- accepted iterations on which the rule
nonetheless saw ρ < η1. `rounding` is the count of accepted iterations whose
predicted reduction sat at or below the rounding level of f.

`predicted` is reconstructed rather than traced. The solver does not expose it,
but on an accepted iteration `f` moves by exactly the achieved reduction, so
`actual_k = obj[k] - obj[k+1]` is exact and `predicted_k = actual_k / ρ_k`
follows from the definition of ρ. Verified against `:obj_trajectory`: a rejected
iteration leaves `obj[k+1] == obj[k]`, and only accepted iterations are counted.
"""
function ab_run(nlp, cfg)
    NLPModels.reset!(nlp)
    p = ab_params(cfg.η1, cfg.frac)
    t0 = time()
    st = tr_solve(nlp; rule = cfg.factory(), model = DEFAULT_MODEL(),
                  subsolver = DEFAULT_SUBSOLVER(), params = p, trace = true)
    dt = time() - t0

    ss  = st.solver_specific
    ρ   = get(ss, :ratio_trajectory,    Float64[])
    acc = get(ss, :accepted_trajectory, Bool[])
    ob  = get(ss, :obj_trajectory,      Float64[])

    n_acc = count(acc)
    band  = count(i -> acc[i] && ρ[i] < cfg.η1, eachindex(acc))

    # The retrospective rules are handed rho-tilde in the rho slot and never see
    # rho_k, so `band` counts something those rules do not read. `band_rule`
    # repeats the count on the ratio the rule actually branched on. For the
    # rho_k rules the two coincide, which is the check that this is the same
    # instrument; where they differ, the rule is not comparable with the rest.
    rr = get(ss, :rho_tilde_trajectory, Float64[])
    band_rule = length(rr) == length(acc) ?
                count(i -> acc[i] && rr[i] < cfg.η1, eachindex(acc)) : -1

    # Accepted iterations whose predicted reduction sat at the rounding level of
    # f. `length(ob) == length(acc) + 1` holds for every deterministic run: the
    # state trajectory carries the initial point, the per-iteration one does not.
    rounding = 0
    if length(ob) == length(acc) + 1
        for i in eachindex(acc)
            acc[i] || continue
            actual = ob[i] - ob[i + 1]
            (isfinite(ρ[i]) && ρ[i] != 0) || continue
            pred = actual / ρ[i]
            abs(pred) <= eps(Float64) * max(1.0, abs(ob[i])) && (rounding += 1)
        end
    end

    return (status    = st.status,
            iter      = st.iter,
            f_evals   = neval_obj(nlp),
            g_evals   = neval_grad(nlp),
            h_evals   = neval_hprod(nlp),
            time      = dt,
            grad      = Float64(st.dual_feas),
            obj       = Float64(st.objective),
            solution  = Vector{Float64}(st.solution),
            n_iter    = length(acc),
            n_acc     = n_acc,
            band      = band,
            band_rule = band_rule,
            rounding  = rounding)
end

"Whether the run reached a critical point. `:second_order` counts, as elsewhere."
ab_solved(r) = r.status in (:first_order, :second_order)

# -----------------------------------------------------------------------------
# The problem set
# -----------------------------------------------------------------------------

"""
    ab_problems() -> (problems, note)

CUTEst, unconstrained, `n ≤ 200`. Never the analytic fallback.

`default_problems()` returns `analytic_problems()` when the CUTEst query comes
back empty, so a run where CUTEst failed to load produces a table that reads as a
CUTEst benchmark and is not one. This stops instead.

`n ≤ 200` rather than config.jl's `MAX_VAR`, so that `SteihaugCG` is not the only
legal subsolver by default and the realised set is the 185 problems Section 5 of
Part III reports. `ExactMS` is admissible on all of them; `SteihaugCG` is used
throughout so that no subsolver switch is confounded with the swept threshold.
"""
function ab_problems()
    HAS_CUTEST || error("exp14 needs CUTEst. `default_problems()` would fall back " *
                        "to the analytic set and the tables would read as a CUTEst " *
                        "benchmark without being one. Install CUTEst and rerun.")
    ps = cutest_problems(min_var = 2, max_var = 200, max_con = 0, limit = nothing)
    isempty(ps) && error("CUTEst loaded but selected no problems at " *
                         "min_var = 2, max_var = 200, max_con = 0.")
    lim = tryparse(Int, get(ENV, "TRR_AB_LIMIT", ""))
    note = ""
    if lim !== nothing && lim < length(ps)
        note = "PILOT: first $lim of $(length(ps)) problems (TRR_AB_LIMIT)"
        ps = ps[1:lim]
    end
    return ps, note
end

# -----------------------------------------------------------------------------
# Reductions
# -----------------------------------------------------------------------------

"Rows for one stratum and rule, indexed by η fraction, over all problems."
ab_select(rows, strat, rule, frac) =
    [r for r in rows if r.strat == strat && r.rule == rule && r.frac == frac]

"""
    ab_paired(rows, strat, rule, frac, field) -> (ratios, wins, losses, ties, only_a, only_b)

Pair `η = frac * η1` against the baseline `η = η1` on the same problem.

A pair enters the log ratio only when both settings solved it. Problems solved by
exactly one setting are returned by name instead, because a log ratio over
survivors would silently drop exactly the cases that matter most.
"""
function ab_paired(rows, strat, rule, frac, field)
    a = Dict(r.problem => r for r in ab_select(rows, strat, rule, frac))
    b = Dict(r.problem => r for r in ab_select(rows, strat, rule, 1.0))
    ratios = Float64[]; wins = 0; losses = 0; ties = 0
    only_a = String[];  only_b = String[]
    for pname in sort(collect(intersect(keys(a), keys(b))))
        ra, rb = a[pname], b[pname]
        sa, sb = ab_solved(ra), ab_solved(rb)
        sa && !sb && (push!(only_a, pname); continue)
        sb && !sa && (push!(only_b, pname); continue)
        (sa && sb) || continue
        va, vb = getfield(ra, field), getfield(rb, field)
        (va > 0 && vb > 0) || continue
        lr = log(va / vb)
        push!(ratios, lr)
        abs(lr) <= AB_TIE ? (ties += 1) : (lr < 0 ? (wins += 1) : (losses += 1))
    end
    return (ratios, wins, losses, ties, only_a, only_b)
end

"Did the two settings converge to different points? Compared only when both solved."
function ab_limit_changed(rows, strat, rule, frac; tol = 1e-4)
    a = Dict(r.problem => r for r in ab_select(rows, strat, rule, frac))
    b = Dict(r.problem => r for r in ab_select(rows, strat, rule, 1.0))
    changed = String[]
    for pname in sort(collect(intersect(keys(a), keys(b))))
        ra, rb = a[pname], b[pname]
        (ab_solved(ra) && ab_solved(rb)) || continue
        length(ra.solution) == length(rb.solution) || continue
        d = norm(ra.solution .- rb.solution)
        s = max(1.0, norm(rb.solution))
        d / s > tol && push!(changed, pname)
    end
    return changed
end

"Quantiles of a vector, or NaN where it is empty."
function ab_quantiles(v)
    isempty(v) && return (NaN, NaN, NaN, NaN, NaN)
    s = sort(v)
    q(p) = s[clamp(1 + floor(Int, p * (length(s) - 1)), 1, length(s))]
    return (q(0.0), q(0.25), q(0.5), q(0.75), q(1.0))
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function acceptance_band(; rules = AB_RULES, tag = "acceptance_band")
    arch = ExperimentArchive(tag = tag)
    problems, pilot_note = ab_problems()
    cfgs = ab_configs(rules)

    @info "Experiment 14: $(length(problems)) problems × $(length(cfgs)) configurations"
    isempty(pilot_note) || @warn pilot_note

    save_config(arch; rules = rules,
                models = [DEFAULT_MODEL], subsolvers = [DEFAULT_SUBSOLVER],
                params = SOLVER_PARAMS,
                problem_selection = Dict("min_var" => 2, "max_var" => 200,
                                         "max_con" => 0, "source" => "CUTEst"),
                extra = Dict("experiment"      => "exp14_acceptance_band",
                             "eta_fractions"   => AB_ETA_FRACTIONS,
                             "strata"          => [string(s) for (s, _) in AB_STRATA],
                             "eta1_values"     => [e for (_, e) in AB_STRATA],
                             "tie_band"        => AB_TIE,
                             "n_problems"      => length(problems),
                             "n_configurations"=> length(cfgs),
                             "pilot"           => pilot_note))

    # --- the realised problem set, written beside the results -----------------
    io = IOBuffer()
    @printf(io, "%-24s %6s\n", "problem", "n")
    println(io, "-"^32)
    realised = Tuple{String, Int}[]

    # --- the runs -------------------------------------------------------------
    rows = NamedTuple[]
    refused = NamedTuple[]
    for (pi, (pname, mk)) in enumerate(problems)
        if has_data(arch, pname, "band")
            try
                d = load_data(arch, pname, "band")
                append!(rows, d["rows"])
                push!(realised, (pname, d["n"]))
                @printf(io, "%-24s %6d\n", pname, d["n"])
                @printf("[%3d/%3d] %-20s cached\n", pi, length(problems), pname)
                continue
            catch
                @warn "unreadable cache, recomputing" problem = pname
            end
        end

        nlp = mk()
        n = nlp.meta.nvar
        push!(realised, (pname, n)); @printf(io, "%-24s %6d\n", pname, n)
        prows = NamedTuple[]
        t0 = time()
        for cfg in cfgs
            r = try
                ab_run(nlp, cfg)
            catch err
                err isa InterruptException && rethrow()
                # A constructor refusal is a recorded outcome, not a run failure.
                # `validate_thresholds` rejects η1 = 0 for the step-driven rules;
                # this sweep never sets η1 = 0, so the list should stay empty.
                push!(refused, (problem = pname, label = cfg.label,
                                why = first(replace(sprint(showerror, err),
                                                    '\n' => ' '), 120)))
                (status = :exception, iter = 0, f_evals = 0, g_evals = 0,
                 h_evals = 0, time = 0.0, grad = NaN, obj = NaN,
                 solution = Float64[], n_iter = 0, n_acc = 0, band = 0,
                 band_rule = 0, rounding = 0)
            end
            push!(prows, merge((problem = pname, n = n, strat = cfg.strat,
                                rule = cfg.rule, frac = cfg.frac,
                                η = cfg.η, η1 = cfg.η1), r))
        end
        finalize(nlp)
        append!(rows, prows)
        save_data(arch, data_filename(pname, "band"); rows = prows, n = n)
        @printf("[%3d/%3d] %-20s n=%-5d %5.1f s\n", pi, length(problems), pname,
                n, time() - t0)
        flush(stdout)
    end
    save_table(arch, "exp14_problem_set.txt", String(take!(io)))

    ab_report(arch, rows, refused, problems, rules, pilot_note)
    return arch, rows
end

"""
    ab_report(arch, rows, refused, problems, rules, pilot_note)

Every table and figure. Split out so a finished archive can be re-reported
without rerunning the campaign.
"""
function ab_report(arch, rows, refused, problems, rules, pilot_note)
    rnames = [nm for (nm, _) in rules]
    retro  = Set(nm for (nm, f) in rules if needs_retrospective(f()))

    # ---- 1. occupancy, per rule and η, distribution across problems ----------
    io = IOBuffer()
    println(io, "Band occupancy: accepted iterations with rho < eta1, as a share")
    println(io, "of accepted iterations. Zero at eta = eta1 by construction.")
    println(io)
    @printf(io, "%-8s %-22s %6s %8s %8s %8s %8s %8s %8s\n", "stratum", "rule",
            "eta/e1", "mean", "median", "q75", "max", "share>0", "tot band")
    println(io, "-"^94)
    for (sname, _) in AB_STRATA, rn in rnames, frac in AB_ETA_FRACTIONS
        rs = ab_select(rows, sname, rn, frac)
        isempty(rs) && continue
        sh = [r.n_acc > 0 ? r.band / r.n_acc : 0.0 for r in rs]
        _, _, med, q75, mx = ab_quantiles(sh)
        @printf(io, "%-8s %-22s %6.2f %8.4f %8.4f %8.4f %8.4f %8.3f %8d\n",
                sname, rn, frac, sum(sh) / length(sh), med, q75, mx,
                count(>(0), sh) / length(sh), sum(r.band for r in rs))
    end
    save_table(arch, "exp14_occupancy.txt", String(take!(io)))

    # ---- 2. paired counts on each work measure ------------------------------
    for (field, fname) in ((:f_evals, "fevals"), (:g_evals, "gevals"),
                           (:h_evals, "hprods"), (:iter, "iterations"))
        io = IOBuffer()
        println(io, "Paired against the baseline eta = eta1, same problem, both solved.")
        println(io, "Measure: ", field, ".  Tie band |log ratio| <= ", AB_TIE, ".")
        field === :iter && println(io, "SECONDARY: iteration counts, not work.")
        println(io)
        @printf(io, "%-8s %-22s %6s %6s %6s %6s %6s %9s %9s %9s\n", "stratum",
                "rule", "eta/e1", "win", "loss", "tie", "pairs", "median", "q25", "q75")
        println(io, "-"^100)
        for (sname, _) in AB_STRATA, rn in rnames, frac in AB_ETA_FRACTIONS
            frac == 1.0 && continue
            lr, w, l, t, oa, ob = ab_paired(rows, sname, rn, frac, field)
            isempty(lr) && w + l + t == 0 && isempty(oa) && isempty(ob) && continue
            _, q25, med, q75, _ = ab_quantiles(lr)
            @printf(io, "%-8s %-22s %6.2f %6d %6d %6d %6d %9.4f %9.4f %9.4f\n",
                    sname, rn, frac, w, l, t, length(lr), med, q25, q75)
        end
        save_table(arch, "exp14_paired_$fname.txt", String(take!(io)))
    end

    # ---- 3. problems solved by exactly one setting, named -------------------
    io = IOBuffer()
    println(io, "Problems solved by exactly one setting of the pair (eta, eta1).")
    println(io, "No filtering on success anywhere else in this report.")
    println(io)
    for (sname, _) in AB_STRATA, rn in rnames, frac in AB_ETA_FRACTIONS
        frac == 1.0 && continue
        _, _, _, _, oa, ob = ab_paired(rows, sname, rn, frac, :f_evals)
        (isempty(oa) && isempty(ob)) && continue
        @printf(io, "%-8s %-22s eta/eta1 = %.2f\n", sname, rn, frac)
        isempty(oa) || println(io, "    solved only at this eta : ", join(oa, ", "))
        isempty(ob) || println(io, "    solved only at eta = eta1: ", join(ob, ", "))
    end
    save_table(arch, "exp14_exclusive.txt", String(take!(io)))

    # ---- 4. full status breakdown, no filtering -----------------------------
    io = IOBuffer()
    println(io, "Status breakdown. Every run, no filtering on success.")
    println(io)
    stats = sort(unique(r.status for r in rows), by = string)
    @printf(io, "%-8s %-22s %6s", "stratum", "rule", "eta/e1")
    for s in stats; @printf(io, "%16s", string(s)); end
    println(io); println(io, "-"^(36 + 16length(stats)))
    for (sname, _) in AB_STRATA, rn in rnames, frac in AB_ETA_FRACTIONS
        rs = ab_select(rows, sname, rn, frac)
        isempty(rs) && continue
        @printf(io, "%-8s %-22s %6.2f", sname, rn, frac)
        for s in stats; @printf(io, "%16d", count(r -> r.status === s, rs)); end
        println(io)
    end
    save_table(arch, "exp14_status.txt", String(take!(io)))

    # ---- 5. the eta = 0 column, with and without rounding-level iterations ---
    io = IOBuffer()
    println(io, "The eta = 0 column. `rounding` counts accepted iterations whose")
    println(io, "predicted reduction sat at or below eps * max(1, |f_k|), where rho")
    println(io, "carries no information. `clean` repeats the paired counts over the")
    println(io, "problems with no such iteration.")
    println(io)
    @printf(io, "%-8s %-22s %8s %8s %8s %8s %8s %8s %8s\n", "stratum", "rule",
            "runs", "w/round", "share", "tot round", "win", "loss", "win-clean")
    println(io, "-"^96)
    for (sname, _) in AB_STRATA, rn in rnames
        rs = ab_select(rows, sname, rn, 0.0)
        isempty(rs) && continue
        dirty = Set(r.problem for r in rs if r.rounding > 0)
        _, w, l, _, _, _ = ab_paired(rows, sname, rn, 0.0, :f_evals)
        clean = filter(r -> !(r.problem in dirty), rows)
        _, wc, _, _, _, _ = ab_paired(clean, sname, rn, 0.0, :f_evals)
        @printf(io, "%-8s %-22s %8d %8d %8.3f %8d %8d %8d %8d\n",
                sname, rn, length(rs), length(dirty),
                length(dirty) / length(rs), sum(r.rounding for r in rs), w, l, wc)
    end
    save_table(arch, "exp14_eta0_rounding.txt", String(take!(io)))

    # ---- 6. limit points ----------------------------------------------------
    io = IOBuffer()
    println(io, "Problems where the two settings converged to different points,")
    println(io, "relative distance > 1e-4, both solved.")
    println(io)
    for (sname, _) in AB_STRATA, rn in rnames, frac in AB_ETA_FRACTIONS
        frac == 1.0 && continue
        ch = ab_limit_changed(rows, sname, rn, frac)
        isempty(ch) && continue
        @printf(io, "%-8s %-22s %.2f  %3d: %s\n", sname, rn, frac, length(ch),
                join(first(ch, 12), ", "))
    end
    save_table(arch, "exp14_limit_points.txt", String(take!(io)))

    # ---- 7. retrospective rules, flagged ------------------------------------
    io = IOBuffer()
    println(io, "Rules reading a ratio other than rho_k in the radius branch.")
    println(io, "Acceptance uses rho_k and eta; the branch uses rho-tilde_{k+1} and")
    println(io, "eta1. The band does not mean the same thing for these, and their")
    println(io, "numbers are not comparable with the rest.")
    println(io)
    @printf(io, "%-22s %-16s %12s %12s
", "rule", "branch ratio",
            "band(rho_k)", "band(rule)")
    println(io, "-"^66)
    for (sname, _) in AB_STRATA, rn in rnames, frac in AB_ETA_FRACTIONS
        rs = ab_select(rows, sname, rn, frac)
        isempty(rs) && continue
        frac == 0.0 || continue
        @printf(io, "%-22s %-16s %12d %12d
", rn,
                rn in retro ? "rho-tilde" : "rho_k",
                sum(r.band for r in rs), sum(r.band_rule for r in rs))
    end
    println(io)
    println(io, "Read at eta = 0 only. Where the two columns differ the rule did")
    println(io, "not branch on the ratio that decided acceptance. RRTR and")
    println(io, "RRTRGrad also compare rho-tilde against their OWN eta-tilde_1 and")
    println(io, "eta-tilde_2 fields, not against the solver's eta1, so neither")
    println(io, "column is a quantity those two rules read. Their first")
    println(io, "discriminant is the `accepted` flag, which eta alone decides.")
    save_table(arch, "exp14_retrospective.txt", String(take!(io)))

    # ---- 8. refusals --------------------------------------------------------
    io = IOBuffer()
    if isempty(refused)
        println(io, "No configuration was refused at construction.")
        println(io, "`validate_thresholds` needs eta1 > 0 for RStep, RAdaptiveStep")
        println(io, "and RDeltaStep. This sweep varies eta and never sets eta1 = 0,")
        println(io, "so no refusal was expected and none occurred.")
    else
        for r in refused
            @printf(io, "%-20s %-28s %s\n", r.problem, r.label, r.why)
        end
    end
    save_table(arch, "exp14_refused.txt", String(take!(io)))

    ab_figures(arch, rows, rnames)

    finalize_archive(arch; notes = """
        Acceptance decoupled from scaling. eta swept over
        $(AB_ETA_FRACTIONS) times eta1, at eta1 in $([e for (_, e) in AB_STRATA]).
        $(length(problems)) CUTEst problems, unconstrained, n <= 200.
        $(isempty(pilot_note) ? "Full problem set." : pilot_note)

        Read exp14_occupancy.txt FIRST. The band is what the experiment is about,
        and no effect reported anywhere else can exceed the exposure recorded
        there. The occupancy column at eta/eta1 = 1.00 is zero by construction and
        is the check that the instrument works.

        Work means evaluations. exp14_paired_fevals.txt, _gevals.txt and
        _hprods.txt are the primary tables; _iterations.txt is secondary and is
        labelled as such in the file. The gradient is evaluated only on
        acceptance, so lowering eta converts rejected iterations into iterations
        that pay a gradient, and the iteration count moves for reasons the
        evaluation count does not share.

        exp14_eta0_rounding.txt separates the eta = 0 column into the part where
        rho was computed from a predicted reduction above the rounding level of f
        and the part where it was not.
        """)
    return arch
end

# -----------------------------------------------------------------------------
# Figures
#
# Black plus one accent, distinguished by line style and marker as well as by
# colour, so nothing is carried by colour alone and the panels survive greyscale.
# -----------------------------------------------------------------------------

const AB_ACCENT = RGB(0.85, 0.33, 0.10)
const AB_BLACK  = RGB(0.0, 0.0, 0.0)
const AB_STYLES = [:solid, :dash, :dot, :dashdot, :solid, :dash, :dot, :dashdot]
const AB_MARKS  = [:circle, :square, :utriangle, :diamond, :cross, :star5, :xcross, :hexagon]

ab_colour(j) = isodd(j) ? AB_BLACK : AB_ACCENT

function ab_figures(arch, rows, rnames)
    xs = AB_ETA_FRACTIONS

    # --- 1. occupancy against eta, one series per rule, spread shown ----------
    for (sname, η1) in AB_STRATA
        plt = plot(xlabel = "η / η₁", ylabel = "band occupancy, share of accepted",
                   legend = :topright, ylims = (-0.02, 1.02),
                   title = "η₁ = $η1 ($sname)")
        for (j, rn) in enumerate(rnames)
            med = Float64[]; lo = Float64[]; hi = Float64[]
            for frac in xs
                rs = ab_select(rows, sname, rn, frac)
                sh = [r.n_acc > 0 ? r.band / r.n_acc : 0.0 for r in rs]
                _, q25, m, q75, _ = ab_quantiles(sh)
                push!(med, m); push!(lo, m - q25); push!(hi, q75 - m)
            end
            plot!(plt, xs, med; yerror = (lo, hi), label = rn,
                  color = ab_colour(j), linestyle = AB_STYLES[j],
                  marker = AB_MARKS[j], lw = 2, ms = 4)
        end
        savefig_archived(arch, "exp14_occupancy_$sname.pdf", plt)
    end

    # --- 2. paired log ratio of gradient evaluations, one panel per rule ------
    for (sname, η1) in AB_STRATA
        panels = []
        for rn in rnames
            data = [ab_paired(rows, sname, rn, f, :g_evals)[1] for f in xs[1:end-1]]
            p = plot(xlabel = "η / η₁", ylabel = "log ratio, ∇f evals",
                     legend = false, title = rn, titlefontsize = 8)
            hline!(p, [0.0]; color = AB_BLACK, linestyle = :dot, lw = 1)
            for (j, v) in enumerate(data)
                isempty(v) && continue
                _, q25, m, q75, _ = ab_quantiles(v)
                plot!(p, [xs[j], xs[j]], [q25, q75]; color = AB_BLACK, lw = 2,
                      label = "")
                scatter!(p, [xs[j]], [m]; color = AB_ACCENT, marker = :circle,
                         ms = 4, label = "")
            end
            push!(panels, p)
        end
        savefig_archived(arch, "exp14_logratio_$sname.pdf",
                         plot(panels...; layout = (2, 4), size = (1400, 600)))
    end

    # --- 3. the mix of work, stacked, against eta ----------------------------
    for (sname, η1) in AB_STRATA
        labs = String[]; F = Float64[]; G = Float64[]; H = Float64[]
        for rn in rnames, frac in xs
            rs = filter(ab_solved, ab_select(rows, sname, rn, frac))
            isempty(rs) && continue
            push!(labs, @sprintf("%s %.2f", first(rn, 8), frac))
            push!(F, sum(r.f_evals for r in rs))
            push!(G, sum(r.g_evals for r in rs))
            push!(H, sum(r.h_evals for r in rs))
        end
        # Numeric x with explicit ticks: `groupedbar` asserts one x number per
        # row, and a vector of strings does not satisfy it.
        plt = groupedbar(1:length(labs), hcat(F, G, H); bar_position = :stack,
                         label = ["f" "∇f" "∇²f·v"], xrotation = 90,
                         xticks = (1:length(labs), labs), xtickfontsize = 5,
                         color = [AB_BLACK AB_ACCENT RGB(0.6, 0.6, 0.6)],
                         ylabel = "evaluations, total over solved problems",
                         legend = :topleft, size = (1500, 600))
        savefig_archived(arch, "exp14_workmix_$sname.pdf", plt)
    end

    # --- 4. performance profiles on evaluations, extremes only ---------------
    for (sname, η1) in AB_STRATA
        panels = []
        for rn in rnames
            a = Dict(r.problem => r for r in ab_select(rows, sname, rn, 0.0))
            b = Dict(r.problem => r for r in ab_select(rows, sname, rn, 1.0))
            names = sort(collect(intersect(keys(a), keys(b))))
            isempty(names) && continue
            M = fill(Inf, length(names), 2)
            for (i, pn) in enumerate(names)
                ab_solved(a[pn]) && (M[i, 1] = a[pn].f_evals + a[pn].g_evals)
                ab_solved(b[pn]) && (M[i, 2] = b[pn].f_evals + b[pn].g_evals)
            end
            all(!isfinite, M) && continue
            τ, prof = performance_profile(M)
            p = plot(τ, prof; xscale = :log10, ylims = (0, 1.02),
                     label = ["η = 0" "η = η₁"], title = rn, titlefontsize = 8,
                     color = [AB_BLACK AB_ACCENT], linestyle = [:solid :dash],
                     lw = 2, xlabel = "τ", ylabel = "π(τ)", legend = :bottomright)
            push!(panels, p)
        end
        isempty(panels) && continue
        savefig_archived(arch, "exp14_profile_$sname.pdf",
                         plot(panels...; layout = (2, 4), size = (1400, 600)))
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    acceptance_band()
end
