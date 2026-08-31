# =============================================================================
# report/task10-second-order-profiles/make_tables.jl
#
# Emits numbers.tex and every tab_*.tex of the report, from the archived runs of
# exp17 and from nothing else. No number in the report is typed by hand.
#
#   julia --project=benchmark report/task10-second-order-profiles/make_tables.jl \
#         benchmark/results/exp_2026-08-30_12-21-55_second_order_profiles
#
# The archive path may also be given as ARCHIVE in the environment. With neither,
# the newest `*_second_order_profiles` directory under benchmark/results is used.
# =============================================================================

using JLD2, Printf, Statistics

const HERE = @__DIR__
const ROOT = normpath(joinpath(HERE, "..", ".."))

function pick_archive()
    length(ARGS) >= 1 && return abspath(ARGS[1])
    haskey(ENV, "ARCHIVE") && return abspath(ENV["ARCHIVE"])
    res = joinpath(ROOT, "benchmark", "results")
    cands = filter(d -> endswith(d, "_second_order_profiles") &&
                        isdir(joinpath(res, d)), readdir(res))
    isempty(cands) && error("make_tables: no *_second_order_profiles archive under $res")
    return joinpath(res, last(sort(cands)))
end

const ARCH = pick_archive()
const DATA = joinpath(ARCH, "data")
isdir(DATA) || error("make_tables: $DATA is not a directory")

# ---- the tolerances and the dense/exact boundary, as exp17 fixed them --------
const EPS_G = 1e-5
const EPS_H = 1e-6
const NMAX  = 200

const RULES = ["Rdelta", "Rstep", "Rrtr", "Rdfo", "Rgrad", "Rgrtr"]
const TEX   = Dict("Rdelta" => "\\Rdelta", "Rstep" => "\\Rstep", "Rrtr" => "\\Rrtr",
                   "Rdfo"   => "\\Rdfo",   "Rgrad" => "\\Rgrad", "Rgrtr" => "\\Rgrtr")

# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------

function load_rows(dir)
    rows = NamedTuple[]
    for f in sort(readdir(dir))
        endswith(f, ".jld2") || continue
        d = try
            JLD2.load(joinpath(dir, f))
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
                     wall = d["wall"], k0 = d["k0_first_order"]))
    end
    return rows
end

const ROWS = load_rows(DATA)
isempty(ROWS) && error("make_tables: no readable runs in $DATA")

const IDX  = Dict((r.problem, r.rule, r.arm, r.pass) => r for r in ROWS)
const DIMS = Dict{String,Int}()
for r in ROWS; DIMS[r.problem] = r.n; end
const PS    = sort(collect(keys(DIMS)))
const SMALL = [p for p in PS if 0 < DIMS[p] <= NMAX]
const BIG   = [p for p in PS if DIMS[p] > NMAX]

# The names the campaign queried, from the realised problem list it wrote. Those
# absent from the archive never opened, and are the six of finding D4.
function listed_problems()
    f = joinpath(ARCH, "tables", "exp17_problems.txt")
    isfile(f) || return String[]
    out = String[]
    for ln in eachline(f)
        m = match(r"^(\S+)\s+(\d+)\s*$", ln)
        m === nothing && continue
        m.captures[1] == "problem" && continue
        push!(out, m.captures[1])
    end
    return out
end
const LISTED  = listed_problems()
const NEVEROP = sort(setdiff(LISTED, PS))

# -----------------------------------------------------------------------------
# Scoring -- the same two criteria exp17 applies, restated here
# -----------------------------------------------------------------------------

ok_first(r)  = isfinite(r.gnorm) && r.gnorm <= EPS_G
ok_second(r) = isfinite(r.gnorm) && isfinite(r.lam) &&
               r.gnorm <= EPS_G && r.lam >= -EPS_H

get_run(p, rn, arm, pass) = get(IDX, (p, rn, arm, pass), nothing)

"""
    rho_inf(ps, rn, arm, pass, ok) -> Float64

The robustness of a profile line: the fraction of `ps` on which `ok` holds.
"""
function rho_inf(ps, rn, arm, pass, ok)
    n = count(ps) do p
        r = get_run(p, rn, arm, pass)
        r !== nothing && ok(r)
    end
    return n / length(ps)
end

"""
    total(ps, rn, arm, pass, metric) -> Int

Summed over every problem, met or not: no filtering on success and no mean over
survivors.
"""
function total(ps, rn, arm, pass, metric)
    s = 0
    for p in ps
        r = get_run(p, rn, arm, pass)
        r === nothing && continue
        s += getproperty(r, metric)
    end
    return s
end

function total_wall(ps, rn, arm, pass)
    s = 0.0
    for p in ps
        r = get_run(p, rn, arm, pass)
        r === nothing && continue
        s += r.wall
    end
    return s
end

nmet(ps, rn, arm, pass, ok) = count(ps) do p
    r = get_run(p, rn, arm, pass); r !== nothing && ok(r)
end

nstatus(rn, arm, pass, st) = count(PS) do p
    r = get_run(p, rn, arm, pass); r !== nothing && r.status === st
end

# -----------------------------------------------------------------------------
# Formatting
# -----------------------------------------------------------------------------

"""
    grp(n) -> String

An integer with thin-space thousands groups, so a column of counts reads at a
glance in print: `1426641` becomes `1\\,426\\,641`.
"""
function grp(n::Integer)
    s = string(abs(n)); out = ""
    while length(s) > 3
        out = "\\," * s[end-2:end] * out
        s = s[1:end-3]
    end
    return (n < 0 ? "-" : "") * s * out
end

sci(x) = isfinite(x) ? replace(@sprintf("%.2e", x), r"e([+-])0*(\d+)" =>
                               s"\\cdot 10^{\1\2}") : "---"
"""
    sig(x) -> String

A signed exponent with no redundant `+`: `-3.77\\cdot 10^{-5}`.
"""
sig(x) = replace(sci(x), "10^{+" => "10^{")

pct(x) = @sprintf("%.1f", 100x)
f2(x)  = @sprintf("%.2f", x)
f3(x)  = @sprintf("%.3f", x)

function write_table(name, body)
    open(joinpath(HERE, name), "w") do io
        print(io, body)
    end
    println("  wrote ", name)
end

# =============================================================================
# tab_campaign.tex -- what actually ran
# =============================================================================

let io = IOBuffer()
    println(io, "\\begin{tabular}{@{}lrr@{}}")
    println(io, "\\toprule")
    println(io, "& first-order arm & second-order arm \\\\")
    println(io, "\\midrule")
    for (lbl, st) in (("\\texttt{first\\_order}", :first_order),
                      ("\\texttt{second\\_order}", :second_order),
                      ("\\texttt{user} (pass~g callback)", :user),
                      ("\\texttt{max\\_iter}", :max_iter),
                      ("\\texttt{max\\_time}", :max_time),
                      ("\\texttt{stalled}", :stalled),
                      ("\\texttt{exception}", :exception))
        a = count(r -> r.arm === :fo && r.status === st, ROWS)
        b = count(r -> r.arm === :so && r.status === st, ROWS)
        @printf(io, "%s & %s & %s \\\\\n", lbl, grp(a), grp(b))
    end
    println(io, "\\midrule")
    @printf(io, "runs & %s & %s \\\\\n",
            grp(count(r -> r.arm === :fo, ROWS)), grp(count(r -> r.arm === :so, ROWS)))
    @printf(io, "runs with a finite \$\\lambda_{\\min}\$ & %s & %s \\\\\n",
            grp(count(r -> r.arm === :fo && isfinite(r.lam), ROWS)),
            grp(count(r -> r.arm === :so && isfinite(r.lam), ROWS)))
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    write_table("tab_campaign.tex", String(take!(io)))
end

# =============================================================================
# tab_p1.tex -- P1, second-order arm on the second-order criterion
# =============================================================================

let io = IOBuffer()
    println(io, "\\begin{tabular}{@{}lrrrrrrr@{}}")
    println(io, "\\toprule")
    println(io, "rule & met & \$\\rho(\\infty)\$ & \$f\$ & \$g\$ & \$Hv\$ & \$H\$ & wall (s) \\\\")
    println(io, "\\midrule")
    for rn in RULES
        @printf(io, "%s & %d & %s & %s & %s & %s & %s & %.0f \\\\\n",
                TEX[rn], nmet(PS, rn, :so, :h, ok_second),
                f3(rho_inf(PS, rn, :so, :h, ok_second)),
                grp(total(PS, rn, :so, :h, :f_evals)),
                grp(total(PS, rn, :so, :h, :g_evals)),
                grp(total(PS, rn, :so, :h, :h_prods)),
                grp(total(PS, rn, :so, :h, :h_evals)),
                total_wall(PS, rn, :so, :h))
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    write_table("tab_p1.tex", String(take!(io)))
end

# =============================================================================
# tab_p2.tex -- P2, both arms on the first-order criterion
# =============================================================================

let io = IOBuffer()
    println(io, "\\begin{tabular}{@{}lrrrrrrr@{}}")
    println(io, "\\toprule")
    println(io, " & \\multicolumn{2}{c}{met} & \\multicolumn{2}{c}{\$f\$ evaluations}")
    println(io, " & \\multicolumn{2}{c}{\$Hv\$ products} & \$H\$ evaluations \\\\")
    println(io, "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-8}")
    println(io, "rule & 1st & 2nd & 1st & 2nd & 1st & 2nd & 2nd \\\\")
    println(io, "\\midrule")
    for rn in RULES
        @printf(io, "%s & %d & %d & %s & %s & %s & %s & %s \\\\\n", TEX[rn],
                nmet(PS, rn, :fo, :g, ok_first), nmet(PS, rn, :so, :g, ok_first),
                grp(total(PS, rn, :fo, :g, :f_evals)), grp(total(PS, rn, :so, :g, :f_evals)),
                grp(total(PS, rn, :fo, :g, :h_prods)), grp(total(PS, rn, :so, :g, :h_prods)),
                grp(total(PS, rn, :so, :g, :h_evals)))
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    write_table("tab_p2.tex", String(take!(io)))
end

# =============================================================================
# tab_p3.tex -- P3, both arms on the second-order criterion, n <= 200
# =============================================================================

let io = IOBuffer()
    println(io, "\\begin{tabular}{@{}lrrrr@{}}")
    println(io, "\\toprule")
    println(io, " & \\multicolumn{2}{c}{met, of $(length(SMALL))}")
    println(io, " & \\multicolumn{2}{c}{runs with a finite \$\\lambda_{\\min}\$} \\\\")
    println(io, "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}")
    println(io, "rule & 1st & 2nd & 1st & 2nd \\\\")
    println(io, "\\midrule")
    for rn in RULES
        nfo = count(p -> (r = get_run(p, rn, :fo, :h); r !== nothing && isfinite(r.lam)), SMALL)
        nso = count(p -> (r = get_run(p, rn, :so, :h); r !== nothing && isfinite(r.lam)), SMALL)
        @printf(io, "%s & %d & %d & %d & %d \\\\\n", TEX[rn],
                nmet(SMALL, rn, :fo, :h, ok_second), nmet(SMALL, rn, :so, :h, ok_second),
                nfo, nso)
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    write_table("tab_p3.tex", String(take!(io)))
end

# =============================================================================
# tab_curv.tex -- curvature at the first-order point (the salvage of P3)
# =============================================================================

curv_stats = Dict{String,NamedTuple}()
let io = IOBuffer()
    println(io, "\\begin{tabular}{@{}lrrrrr@{}}")
    println(io, "\\toprule")
    println(io, "rule & reached & \$\\lambda_{\\min} \\geq -\\varepsilon_H\$ & " *
                "\$\\lambda_{\\min} < -\\varepsilon_H\$ & share & median \$\\lambda_{\\min}\$ when negative \\\\")
    println(io, "\\midrule")
    for rn in RULES
        sel = NamedTuple[]
        for p in SMALL
            r = get_run(p, rn, :so, :g)
            r === nothing && continue
            (ok_first(r) && isfinite(r.lam)) || continue
            push!(sel, r)
        end
        neg = [r.lam for r in sel if r.lam < -EPS_H]
        curv_stats[rn] = (reached = length(sel), pos = length(sel) - length(neg),
                          neg = length(neg),
                          share = length(neg) / max(1, length(sel)),
                          med = isempty(neg) ? NaN : median(neg))
        @printf(io, "%s & %d & %d & %d & %s\\%% & \$%s\$ \\\\\n", TEX[rn],
                length(sel), length(sel) - length(neg), length(neg),
                pct(length(neg) / max(1, length(sel))),
                isempty(neg) ? "\\text{---}" : sig(median(neg)))
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    write_table("tab_curv.tex", String(take!(io)))
end

# =============================================================================
# tab_extra.tex -- what certifying curvature cost, pass h against pass g
# =============================================================================

let io = IOBuffer()
    println(io, "\\begin{tabular}{@{}lrrrrr@{}}")
    println(io, "\\toprule")
    println(io, "rule & problems & \$f\$ & \$g\$ & iterations & \$H\$ \\\\")
    println(io, "\\midrule")
    for rn in RULES
        fr = Float64[]; gr = Float64[]; ir = Float64[]; hr = Float64[]
        for p in SMALL
            a = get_run(p, rn, :so, :g); b = get_run(p, rn, :so, :h)
            (a === nothing || b === nothing) && continue
            (ok_first(a) && ok_second(b)) || continue
            a.f_evals > 0 && push!(fr, b.f_evals / a.f_evals)
            a.g_evals > 0 && push!(gr, b.g_evals / a.g_evals)
            a.iters   > 0 && push!(ir, b.iters   / a.iters)
            a.h_evals > 0 && push!(hr, b.h_evals / a.h_evals)
        end
        @printf(io, "%s & %d & %s & %s & %s & %s \\\\\n", TEX[rn], length(fr),
                f3(median(fr)), f3(median(gr)), f3(median(ir)), f3(median(hr)))
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    write_table("tab_extra.tex", String(take!(io)))
end

# =============================================================================
# tab_obj.tex -- where the two arms landed
# =============================================================================

obj_stats = Dict{String,NamedTuple}()
let io = IOBuffer()
    println(io, "\\begin{tabular}{@{}lrrrrr@{}}")
    println(io, "\\toprule")
    println(io, "rule & compared & 2nd lower & 1st lower & same point & median gap \\\\")
    println(io, "\\midrule")
    for rn in RULES
        nb = nf = nt = 0; gaps = Float64[]
        for p in SMALL
            a = get_run(p, rn, :fo, :h); b = get_run(p, rn, :so, :h)
            (a === nothing || b === nothing) && continue
            ok_first(a) || continue
            ok_second(b) || continue
            (isfinite(a.obj) && isfinite(b.obj)) || continue
            sc = max(1.0, abs(a.obj), abs(b.obj))
            rel = (a.obj - b.obj) / sc
            if rel > 1e-8
                nb += 1; push!(gaps, rel)
            elseif rel < -1e-8
                nf += 1; push!(gaps, rel)
            else
                nt += 1
            end
        end
        obj_stats[rn] = (cmp = nb + nf + nt, so = nb, fo = nf, tie = nt,
                         med = isempty(gaps) ? NaN : median(gaps))
        @printf(io, "%s & %d & %d & %d & %d & \$%s\$ \\\\\n", TEX[rn],
                nb + nf + nt, nb, nf, nt,
                isempty(gaps) ? "\\text{---}" : sig(median(gaps)))
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    write_table("tab_obj.tex", String(take!(io)))
end

# =============================================================================
# tab_gaps.tex -- the (problem, rule) pairs where the arms genuinely disagreed
# =============================================================================

gaps_all = Tuple[]
for p in SMALL, rn in RULES
    a = get_run(p, rn, :fo, :h); b = get_run(p, rn, :so, :h)
    (a === nothing || b === nothing) && continue
    ok_first(a) || continue
    ok_second(b) || continue
    (isfinite(a.obj) && isfinite(b.obj)) || continue
    sc = max(1.0, abs(a.obj), abs(b.obj))
    rel = (a.obj - b.obj) / sc
    abs(rel) > 1e-4 && push!(gaps_all, (abs(rel), p, rn, DIMS[p], a.obj, b.obj, rel))
end
sort!(gaps_all, rev = true)

let io = IOBuffer()
    # One line per (problem, rule) pair. A pair is the unit here: on MGH17SLS and
    # BIGGS6 the direction differs between rules, so a per-problem row would have
    # to average two opposite findings and would state neither.
    println(io, "\\begin{tabular}{@{}lrlrrl@{}}")
    println(io, "\\toprule")
    println(io, "problem & \$n\$ & rule & \$f\$, 1st arm & \$f\$, 2nd arm & \\\\")
    println(io, "\\midrule")
    for g in gaps_all[1:min(16, length(gaps_all))]
        @printf(io, "\\texttt{%s} & %d & %s & \$%s\$ & \$%s\$ & %s \\\\\n",
                replace(g[2], "_" => "\\_"), g[4], TEX[g[3]], sig(g[5]), sig(g[6]),
                g[7] > 0 ? "2nd lower" : "1st lower")
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    write_table("tab_gaps.tex", String(take!(io)))
end

# =============================================================================
# numbers.tex -- every number the prose quotes
# =============================================================================

let io = IOBuffer()
    pr(k, v) = println(io, "\\newcommand{\\", k, "}{", v, "}")

    println(io, "% Generated by make_tables.jl from")
    println(io, "%   ", replace(basename(ARCH), "_" => "\\_"))
    println(io, "% Do not edit by hand.")
    println(io)

    # ---- the campaign
    pr("NListed",  length(LISTED))
    pr("NOpened",  length(PS))
    pr("NNeverOp", length(NEVEROP))
    pr("NeverOpened", join(["\\texttt{" * replace(p, "_" => "\\_") * "}" for p in NEVEROP], ", "))
    pr("NSmall",   length(SMALL))
    pr("NBig",     length(BIG))
    pr("NRules",   length(RULES))
    pr("NRuns",    grp(length(ROWS)))
    pr("NRunsPerProblem", length(RULES) * 4)
    pr("EpsG", "10^{-5}")
    pr("EpsH", "10^{-6}")
    pr("NMaxExact", NMAX)

    # ---- exceptions: the six that never opened against the ones that raised
    exc = [r for r in ROWS if r.status === :exception]
    excp = sort(unique(r.problem for r in exc))
    pr("NExcArchived", length(exc))
    pr("NExcReported", length(exc) + length(NEVEROP) * length(RULES) * 4)
    pr("NExcPhantom",  length(NEVEROP) * length(RULES) * 4)
    pr("ExcProblems",  join(["\\texttt{" * replace(p, "_" => "\\_") * "}" for p in excp], ", "))
    pr("ExcDim", isempty(excp) ? 0 : DIMS[excp[1]])

    # ---- lambda_min availability, the D1 evidence
    for (nm, arm) in (("Fo", :fo), ("So", :so))
        pr("NLam$(nm)Finite", grp(count(r -> r.arm === arm && isfinite(r.lam), ROWS)))
        pr("NLam$(nm)Total",  grp(count(r -> r.arm === arm, ROWS)))
    end

    # ---- P1 headline
    met1 = Dict(rn => nmet(PS, rn, :so, :h, ok_second) for rn in RULES)
    pr("PoneMetMin", minimum(values(met1)))
    pr("PoneMetMax", maximum(values(met1)))
    for rn in RULES
        pr("Pone$(rn)Met",  met1[rn])
        pr("Pone$(rn)Rho",  f3(rho_inf(PS, rn, :so, :h, ok_second)))
        pr("Pone$(rn)F",    grp(total(PS, rn, :so, :h, :f_evals)))
        pr("Pone$(rn)G",    grp(total(PS, rn, :so, :h, :g_evals)))
        pr("Pone$(rn)Hp",   grp(total(PS, rn, :so, :h, :h_prods)))
        pr("Pone$(rn)He",   grp(total(PS, rn, :so, :h, :h_evals)))
    end
    fg(rn, m) = total(PS, rn, :so, :h, m)
    pr("GrtrVsGradF",  pct(1 - fg("Rgrtr", :f_evals) / fg("Rgrad", :f_evals)))
    pr("GrtrVsGradHe", pct(1 - fg("Rgrtr", :h_evals) / fg("Rgrad", :h_evals)))
    pr("GrtrVsDfoHe",  pct(1 - fg("Rgrtr", :h_evals) / fg("Rdfo",  :h_evals)))
    pr("DfoVsGrtrF",   f2(fg("Rdfo", :f_evals) / fg("Rgrtr", :f_evals)))
    pr("DfoVsGrtrHe",  f2(fg("Rdfo", :h_evals) / fg("Rgrtr", :h_evals)))

    # ---- P2 headline
    dmin, dmax = 10^6, -10^6
    rmin, rmax = 1e9, -1e9
    for rn in RULES
        d = nmet(PS, rn, :so, :g, ok_first) - nmet(PS, rn, :fo, :g, ok_first)
        dmin = min(dmin, d); dmax = max(dmax, d)
        ratio = total(PS, rn, :fo, :g, :h_prods) / total(PS, rn, :so, :g, :h_prods)
        rmin = min(rmin, ratio); rmax = max(rmax, ratio)
        pr("Ptwo$(rn)Fo", nmet(PS, rn, :fo, :g, ok_first))
        pr("Ptwo$(rn)So", nmet(PS, rn, :so, :g, ok_first))
    end
    pr("PtwoGainMin", dmin)
    pr("PtwoGainMax", dmax)
    pr("PtwoHpMin",   f2(rmin))
    pr("PtwoHpMax",   f2(rmax))
    pr("PtwoStepFo",  nmet(PS, "Rstep", :fo, :g, ok_first))
    pr("PtwoStepSo",  nmet(PS, "Rstep", :so, :g, ok_first))
    pr("PtwoStepFRatio", f2(total(PS, "Rstep", :fo, :g, :f_evals) /
                            total(PS, "Rstep", :so, :g, :f_evals)))
    pr("PtwoRtrFRatio",  f2(total(PS, "Rrtr", :so, :g, :f_evals) /
                            total(PS, "Rrtr", :fo, :g, :f_evals)))
    pr("PtwoDeltaHpSaved", grp(total(PS, "Rdelta", :fo, :g, :h_prods) -
                               total(PS, "Rdelta", :so, :g, :h_prods)))
    pr("PtwoDeltaHe", grp(total(PS, "Rdelta", :so, :g, :h_evals)))

    # ---- stalling, pass g
    for (nm, arm) in (("Fo", :fo), ("So", :so))
        pr("Stall$(nm)", count(r -> r.arm === arm && r.pass === :g && r.status === :stalled, ROWS))
        pr("MaxIter$(nm)", count(r -> r.arm === arm && r.pass === :g && r.status === :max_iter, ROWS))
        pr("MaxTime$(nm)", count(r -> r.arm === arm && r.pass === :g && r.status === :max_time, ROWS))
    end

    # ---- the salvage
    shares = [curv_stats[rn].share for rn in RULES]
    pr("CurvShareMin", pct(minimum(shares)))
    pr("CurvShareMax", pct(maximum(shares)))
    pr("CurvAlreadyMin", pct(1 - maximum(shares)))
    pr("CurvAlreadyMax", pct(1 - minimum(shares)))
    pr("CurvNegMin", minimum(curv_stats[rn].neg for rn in RULES))
    pr("CurvNegMax", maximum(curv_stats[rn].neg for rn in RULES))
    pr("ObjSoMin", minimum(obj_stats[rn].so for rn in RULES))
    pr("ObjSoMax", maximum(obj_stats[rn].so for rn in RULES))
    pr("ObjFoMin", minimum(obj_stats[rn].fo for rn in RULES))
    pr("ObjFoMax", maximum(obj_stats[rn].fo for rn in RULES))
    pr("NGapPairs", length(gaps_all))
    pr("NComparedPairs", length(SMALL) * length(RULES))

    # ---- P3
    pr("PthreeSoMin", minimum(nmet(SMALL, rn, :so, :h, ok_second) for rn in RULES))
    pr("PthreeSoMax", maximum(nmet(SMALL, rn, :so, :h, ok_second) for rn in RULES))

    write_table("numbers.tex", String(take!(io)))
end

println("\nArchive: ", ARCH)
@printf("%d runs, %d problems opened of %d listed, %d with n <= %d.\n",
        length(ROWS), length(PS), length(LISTED), length(SMALL), NMAX)
isempty(NEVEROP) || println("Never opened: ", join(NEVEROP, ", "))
