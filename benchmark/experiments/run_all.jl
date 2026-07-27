# =============================================================================
# benchmark/experiments/run_all.jl
#
# Run every experiment in sequence, each into its own timestamped archive.
#
#   julia --project=benchmark benchmark/experiments/run_all.jl
#   julia --project=benchmark benchmark/experiments/run_all.jl 1 3 5
# =============================================================================

const EXPERIMENTS = [
    ("1", "exp1_comparison.jl",       "comparison of mechanisms"),
    ("2", "exp2_trajectories.jl",     "per-iteration trajectories"),
    ("3", "exp3_zeta_sweep.jl",       "influence of ζ"),
    ("4", "exp4_mu_sweep.jl",         "influence of the cap μ_max"),
    ("5", "exp5_inactivity.jl",       "trust-region inactivity"),
    ("6", "exp6_interaction.jl",      "mechanism × model Hessian"),
    ("7", "exp7_convergence_rate.jl", "local convergence order"),
]

function main(args)
    wanted = isempty(args) ? [e[1] for e in EXPERIMENTS] : args
    for (id, file, desc) in EXPERIMENTS
        id in wanted || continue
        println("\n", "="^70)
        println("EXPERIMENT $id -- $desc")
        println("="^70)
        try
            include(joinpath(@__DIR__, file))
        catch err
            err isa InterruptException && rethrow()
            @error "experiment $id failed" err
        end
    end
    println("\nAll requested experiments finished. Archives are under benchmark/results/.")
end

main(ARGS)
