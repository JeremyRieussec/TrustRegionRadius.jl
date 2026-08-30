# =============================================================================
# _audit/mu_theta.jl
#
# The median of theta_k = Delta_k / ||g_k|| over an exp4 archive.
#
# `exp4_climb.txt` records how many iterations had an expansion clipped by the
# cap. That counts requests refused, not time spent at the cap: an iteration
# that contracts moves mu_k strictly below mu_bar and is never counted, so the
# clipped count is a lower bound on the time at the cap rather than a measure of
# it. theta_k is the direct measure. RGradCapped sets Delta_k = mu_k ||g_k||, so
# theta_k = mu_k and theta_k = mu_bar exactly when the multiplier sits at the cap.
#
#   julia --project=benchmark _audit/mu_theta.jl <archive directory>
# =============================================================================
using JLD2, Printf, Statistics

archdir = length(ARGS) >= 1 ? ARGS[1] :
          error("usage: mu_theta.jl <archive directory>")
datadir = joinpath(archdir, "data")
isdir(datadir) || error("no data/ under $archdir")

"mu_bar from a configuration name such as `mu_max=0.001`; `nothing` if uncapped."
cap_of(name) = startswith(name, "mu_max=") ?
               parse(Float64, split(name, '=')[2]) : nothing

rows = NamedTuple[]
for f in sort(readdir(datadir))
    endswith(f, ".jld2") || continue
    d = JLD2.load(joinpath(datadir, f))
    cfg = d["rule_name"]
    Δ, g = d["delta_trajectory"], d["grad_norm_trajectory"]
    br = get(d, "branch_trajectory", Symbol[])
    (isempty(Δ) || isempty(g)) && continue

    # theta on the iterations where it is defined. Both trajectories carry the
    # initial point, so they have the same length and pair index by index.
    n = min(length(Δ), length(g))
    θ = [Δ[i] / g[i] for i in 1:n if isfinite(g[i]) && g[i] > 0 && isfinite(Δ[i])]
    isempty(θ) && continue

    μ̄ = cap_of(cfg)
    # An iteration sits AT the cap when theta_k reaches mu_bar. The comparison is
    # relative, so that it does not depend on the magnitude of mu_bar.
    at_cap = μ̄ === nothing ? NaN :
             count(t -> t >= μ̄ * (1 - 1e-9), θ) / length(θ)
    push!(rows, (config = cfg, problem = d["problem_name"],
                 mubar = μ̄, med_theta = median(θ), n_theta = length(θ),
                 at_cap = at_cap,
                 capped = count(==(:expand_capped), br)))
end
isempty(rows) && error("no usable runs in $archdir")

cfgs = String[]
for r in rows; r.config in cfgs || push!(cfgs, r.config); end

println("theta_k = Delta_k / ||g_k||, from $(length(rows)) runs in")
println(basename(archdir))
println()
println("`bound` is the run set of Table `tab: mu climb`: at least one expansion")
println("clipped at mu_bar. `med theta/mu_bar` is the median over those runs of")
println("the per-run median theta_k, divided by mu_bar. `at cap` is the median")
println("over those runs of the fraction of iterations with theta_k = mu_bar.")
println()
@printf("%-14s %10s %9s %16s %14s %14s\n", "config", "mu_bar", "bound",
        "med theta/mu_bar", "at cap (med)", "at cap (min)")
println("-"^84)
for c in cfgs
    v = [r for r in rows if r.config == c]
    B = [r for r in v if r.capped >= 1]
    μ̄ = v[1].mubar
    if μ̄ === nothing || isempty(B)
        @printf("%-14s %10s %9s %16s %14s %14s\n", c,
                μ̄ === nothing ? "n/a" : @sprintf("%g", μ̄),
                @sprintf("%d/%d", length(B), length(v)), "n/a", "n/a", "n/a")
        continue
    end
    @printf("%-14s %10g %9s %16.6f %14.4f %14.4f\n", c, μ̄,
            @sprintf("%d/%d", length(B), length(v)),
            median([r.med_theta for r in B]) / μ̄,
            median([r.at_cap for r in B]),
            minimum(r.at_cap for r in B))
end

println()
println("For reference, the same two columns over ALL 196 runs of each")
println("configuration rather than over the cap-bound ones:")
println()
@printf("%-14s %16s %14s\n", "config", "med theta/mu_bar", "at cap (med)")
println("-"^46)
for c in cfgs
    v = [r for r in rows if r.config == c]
    μ̄ = v[1].mubar
    μ̄ === nothing && continue
    @printf("%-14s %16.6f %14.4f\n", c,
            median([r.med_theta for r in v]) / μ̄,
            median([r.at_cap for r in v]))
end
