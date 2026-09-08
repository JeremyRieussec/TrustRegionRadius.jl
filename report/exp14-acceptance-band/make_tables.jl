# =============================================================================
# report/exp14-acceptance-band/make_tables.jl
#
# Turn an exp14 archive into the LaTeX fragments the report inputs.
#
#   julia --project=benchmark report/exp14-acceptance-band/make_tables.jl \
#         benchmark/results/exp_..._acceptance_band
#
# Every number the report states is emitted here, either into a table or into a
# \newcommand in numbers.tex. Nothing is transcribed by hand, so no number in
# the prose can drift from the archive it came from.
#
# No `---` in any cell: that sets an em dash. Empty cells are left empty.
# =============================================================================

include(joinpath(@__DIR__, "..", "..", "benchmark", "initialisation.jl"))
using JLD2, Printf, Statistics

const OUT = @__DIR__

archdir = length(ARGS) >= 1 ? ARGS[1] :
          error("usage: make_tables.jl <archive directory>")
isdir(archdir) || error("no such archive: $archdir")

# --- load ---------------------------------------------------------------------
rows = NamedTuple[]
pset = Tuple{String, Int}[]
for f in sort(readdir(joinpath(archdir, "data")))
    endswith(f, "__band.jld2") || continue
    d = JLD2.load(joinpath(archdir, "data", f))
    append!(rows, d["rows"])
    push!(pset, (replace(f, "__band.jld2" => ""), d["n"]))
end
isempty(rows) && error("no rows in $archdir")
rnames = [nm for (nm, _) in AB_RULES]
@info "loaded" nproblems=length(pset) nrows=length(rows)

esc(s) = replace(String(s), "_" => raw"\_")

# =============================================================================
# 1. occupancy
# =============================================================================
open(joinpath(OUT, "tab_occupancy.tex"), "w") do io
    println(io, raw"\begin{tabular}{@{}ll" * "r"^5 * raw"@{}}")
    println(io, raw"\toprule")
    println(io, raw"& & \multicolumn{5}{c@{}}{$\eta/\eta_1$} \\ ")
    println(io, raw"\cmidrule(l){3-7}")
    println(io, raw"$\eta_1$ & Rule & $0$ & $0.25$ & $0.50$ & $0.75$ & $1$ \\ ")
    println(io, raw"\midrule")
    for (si, (sname, η1)) in enumerate(AB_STRATA)
        for (j, rn) in enumerate(rnames)
            @printf(io, "%s & %s ", j == 1 ? "\$$η1\$" : "", esc(rn))
            for frac in AB_ETA_FRACTIONS
                rs = ab_select(rows, sname, rn, frac)
                sh = isempty(rs) ? [0.0] :
                     [r.n_acc > 0 ? r.band / r.n_acc : 0.0 for r in rs]
                @printf(io, "& %.3f ", mean(sh))
            end
            println(io, raw"\\ ")
        end
        si < length(AB_STRATA) && println(io, raw"\midrule")
    end
    println(io, raw"\bottomrule")
    println(io, raw"\end{tabular}")
end

# =============================================================================
# 2. paired counts, gradient evaluations (the primary work measure)
# =============================================================================
function paired_table(field, fname)
    open(joinpath(OUT, "tab_paired_$fname.tex"), "w") do io
        println(io, raw"\begin{tabular}{@{}ll" * "rrr"^4 * raw"@{}}")
        println(io, raw"\toprule")
        println(io, raw"& & \multicolumn{3}{c}{$\eta/\eta_1 = 0$}" *
                    raw" & \multicolumn{3}{c}{$0.25$}" *
                    raw" & \multicolumn{3}{c}{$0.50$}" *
                    raw" & \multicolumn{3}{c@{}}{$0.75$} \\ ")
        println(io, raw"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(l){12-14}")
        println(io, raw"$\eta_1$ & Rule " * repeat(raw"& W & L & T ", 4) * raw"\\ ")
        println(io, raw"\midrule")
        for (si, (sname, η1)) in enumerate(AB_STRATA)
            for (j, rn) in enumerate(rnames)
                @printf(io, "%s & %s ", j == 1 ? "\$$η1\$" : "", esc(rn))
                for frac in AB_ETA_FRACTIONS[1:end-1]
                    _, w, l, t, _, _ = ab_paired(rows, sname, rn, frac, field)
                    @printf(io, "& %d & %d & %d ", w, l, t)
                end
                println(io, raw"\\ ")
            end
            si < length(AB_STRATA) && println(io, raw"\midrule")
        end
        println(io, raw"\bottomrule")
        println(io, raw"\end{tabular}")
    end
end
paired_table(:g_evals, "gevals")
paired_table(:f_evals, "fevals")

# =============================================================================
# 3. median log ratio on each work measure, narrow stratum
# =============================================================================
open(joinpath(OUT, "tab_logratio.tex"), "w") do io
    println(io, raw"\begin{tabular}{@{}ll" * "rrrr"^1 * raw"@{}}")
    println(io, raw"\toprule")
    println(io, raw"$\eta_1$ & Rule & $f$ & $\nabla f$ & $\nabla^2 f\,v$ & iterations \\ ")
    println(io, raw"\midrule")
    for (si, (sname, η1)) in enumerate(AB_STRATA)
        for (j, rn) in enumerate(rnames)
            @printf(io, "%s & %s ", j == 1 ? "\$$η1\$" : "", esc(rn))
            for field in (:f_evals, :g_evals, :h_evals, :iter)
                lr, _, _, _, _, _ = ab_paired(rows, sname, rn, 0.0, field)
                _, _, m, _, _ = ab_quantiles(lr)
                isnan(m) ? print(io, "& ") : @printf(io, "& %+.3f ", m)
            end
            println(io, raw"\\ ")
        end
        si < length(AB_STRATA) && println(io, raw"\midrule")
    end
    println(io, raw"\bottomrule")
    println(io, raw"\end{tabular}")
end

# =============================================================================
# 4. status breakdown
# =============================================================================
allstat = sort(unique(string(r.status) for r in rows))
open(joinpath(OUT, "tab_status.tex"), "w") do io
    println(io, raw"\begin{tabular}{@{}lr" * "r"^length(allstat) * raw"@{}}")
    println(io, raw"\toprule")
    println(io, raw"$\eta_1$ & $\eta/\eta_1$ & " *
                join([raw"\texttt{" * esc(s) * "}" for s in allstat], " & ") * raw" \\ ")
    println(io, raw"\midrule")
    for (si, (sname, η1)) in enumerate(AB_STRATA)
        for (j, frac) in enumerate(AB_ETA_FRACTIONS)
            rs = [r for r in rows if r.strat == sname && r.frac == frac &&
                  r.rule in rnames]
            @printf(io, "%s & %.2f ", j == 1 ? "\$$η1\$" : "", frac)
            for s in allstat
                @printf(io, "& %d ", count(r -> string(r.status) == s, rs))
            end
            println(io, raw"\\ ")
        end
        si < length(AB_STRATA) && println(io, raw"\midrule")
    end
    println(io, raw"\bottomrule")
    println(io, raw"\end{tabular}")
end

# =============================================================================
# 5. the eta = 0 column, with and without rounding-level iterations
# =============================================================================
open(joinpath(OUT, "tab_eta0.tex"), "w") do io
    println(io, raw"\begin{tabular}{@{}llrrrrrr@{}}")
    println(io, raw"\toprule")
    println(io, raw"& & \multicolumn{2}{c}{rounding level}" *
                raw" & \multicolumn{2}{c}{all problems}" *
                raw" & \multicolumn{2}{c@{}}{clean only} \\ ")
    println(io, raw"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}")
    println(io, raw"$\eta_1$ & Rule & probs & iters & W & L & W & L \\ ")
    println(io, raw"\midrule")
    for (si, (sname, η1)) in enumerate(AB_STRATA)
        for (j, rn) in enumerate(rnames)
            rs = ab_select(rows, sname, rn, 0.0)
            dirty = Set(r.problem for r in rs if r.rounding > 0)
            _, w, l, _, _, _ = ab_paired(rows, sname, rn, 0.0, :g_evals)
            clean = filter(r -> !(r.problem in dirty), rows)
            _, wc, lc, _, _, _ = ab_paired(clean, sname, rn, 0.0, :g_evals)
            @printf(io, "%s & %s & %d & %d & %d & %d & %d & %d \\\\\n",
                    j == 1 ? "\$$η1\$" : "", esc(rn), length(dirty),
                    sum(r.rounding for r in rs), w, l, wc, lc)
        end
        si < length(AB_STRATA) && println(io, raw"\midrule")
    end
    println(io, raw"\bottomrule")
    println(io, raw"\end{tabular}")
end

# =============================================================================
# 6. the retrospective pair, band on rho_k against band on the rule's ratio
# =============================================================================
open(joinpath(OUT, "tab_retro.tex"), "w") do io
    println(io, raw"\begin{tabular}{@{}llrr@{}}")
    println(io, raw"\toprule")
    println(io, raw"Rule & Branch ratio & band on $\rho_k$ & band on rule ratio \\ ")
    println(io, raw"\midrule")
    for rn in rnames
        rs = ab_select(rows, :narrow, rn, 0.0)
        isempty(rs) && continue
        isretro = any(r -> r.band_rule != r.band, rs)
        @printf(io, "%s & %s & %d & %d \\\\\n", esc(rn),
                isretro ? raw"$\widetilde\rho_{k+1}$" : raw"$\rho_k$",
                sum(r.band for r in rs), sum(r.band_rule for r in rs))
    end
    println(io, raw"\bottomrule")
    println(io, raw"\end{tabular}")
end

# =============================================================================
# 7. numbers quoted in the prose
# =============================================================================
open(joinpath(OUT, "numbers.tex"), "w") do io
    cmd(k, v) = println(io, "\\newcommand{\\", k, "}{", v, "}")
    cmd("NProblems",  length(pset))
    cmd("NRules",     length(rnames))
    cmd("NConfigs",   length(AB_STRATA) * length(rnames) * length(AB_ETA_FRACTIONS))
    cmd("NRuns",      length(rows))
    cmd("NMin",       minimum(p[2] for p in pset))
    cmd("NMax",       maximum(p[2] for p in pset))
    cmd("TieBand",    AB_TIE)

    # the instrument check
    cmd("BandAtBaseline", sum(r.band for r in rows if r.frac == 1.0))

    # occupancy at eta = 0, pooled over the eight rules, per stratum
    for (sname, η1) in AB_STRATA
        rs = [r for r in rows if r.strat == sname && r.frac == 0.0 && r.rule in rnames]
        sh = [r.n_acc > 0 ? r.band / r.n_acc : 0.0 for r in rs]
        tag = uppercasefirst(string(sname))
        cmd("Occ$tag",       @sprintf("%.3f", mean(sh)))
        cmd("OccMed$tag",    @sprintf("%.3f", ab_quantiles(sh)[3]))
        cmd("OccPos$tag",    @sprintf("%.3f", count(>(0), sh) / length(sh)))
        cmd("OccMax$tag",    @sprintf("%.3f", maximum(sh)))
    end

    # paired totals at eta = 0, pooled over the eight rules
    for (sname, _) in AB_STRATA
        W = L = T = 0
        for rn in rnames
            _, w, l, t, _, _ = ab_paired(rows, sname, rn, 0.0, :g_evals)
            W += w; L += l; T += t
        end
        tag = uppercasefirst(string(sname))
        cmd("PairW$tag", W); cmd("PairL$tag", L); cmd("PairT$tag", T)
    end

    # rounding-level exposure at eta = 0
    r0 = [r for r in rows if r.frac == 0.0 && r.rule in rnames]
    cmd("RoundRuns",  count(r -> r.rounding > 0, r0))
    cmd("RoundTotal", length(r0))
    cmd("RoundShare", @sprintf("%.3f", count(r -> r.rounding > 0, r0) / length(r0)))

    # limit points
    ch = 0
    for (sname, _) in AB_STRATA, rn in rnames, frac in AB_ETA_FRACTIONS[1:end-1]
        ch += length(ab_limit_changed(rows, sname, rn, frac))
    end
    cmd("LimitChanged", ch)

    # solve counts at the two extremes, pooled
    for (sname, _) in AB_STRATA, (frac, tag) in ((0.0, "Zero"), (1.0, "Base"))
        rs = [r for r in rows if r.strat == sname && r.frac == frac && r.rule in rnames]
        cmd("Solved" * tag * uppercasefirst(string(sname)), count(ab_solved, rs))
        cmd("Total"  * tag * uppercasefirst(string(sname)), length(rs))
    end
end

@info "wrote LaTeX fragments" dir=OUT
println("done")
