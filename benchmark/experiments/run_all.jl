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
    ("8", "exp8_single_problem.jl",   "single-problem diagnostics"),
    ("9", "exp9_second_order.jl",    "first- vs second-order anchoring"),
    ("10","exp10_stochastic.jl",     "sampling rules under noise"),
    ("11","exp11_bhhh.jl",           "outer-product Hessians: BHHH, BHHH-2, Gauss-Newton"),
    ("12","exp12_sampling_examples.jl", "three examples x every sampling rule"),
]

function main(args)
    wanted = isempty(args) ? [e[1] for e in EXPERIMENTS] : args
    for (id, file, desc) in EXPERIMENTS
        id in wanted || continue
        println("\n", "="^70)
        println("EXPERIMENT $id -- $desc")
        println("="^70)
        try
            # Each experiment file guards its `main()` on being the program file,
            # so including one only *defines* it -- as written, this loop produced
            # empty archives. Call it explicitly, and through `invokelatest`:
            # `main` is defined by the `include` on the line above and so lives in
            # a newer world age than this function's compiled code.
            include(joinpath(@__DIR__, file))
            Base.invokelatest(main)
        catch err
            err isa InterruptException && rethrow()
            @error "experiment $id failed" err
        end
    end
    println("\nAll requested experiments finished. Archives are under benchmark/results/.")
end

main(ARGS)
