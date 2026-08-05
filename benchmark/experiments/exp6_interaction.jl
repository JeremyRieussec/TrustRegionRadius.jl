# =============================================================================
# benchmark/experiments/exp6_interaction.jl
#
# EXPERIMENT 6 -- interaction between the radius mechanism and the model Hessian.
#
# The survey's structural claim is that the radius rule sets the step LENGTH
# while the model sets its DIRECTION, so the limit point should track the model
# and not the rule. This runs the full grid and tests whether the two axes
# interact or merely add.
#
#   julia --project=benchmark benchmark/experiments/exp6_interaction.jl
# =============================================================================

const GRID_RULES = [
    ("RDelta", () -> RDelta()),
    ("RStep",  () -> RStep()),
    ("RDFO",   () -> RDFO(ζ = 1.0)),
    ("RGrad",  () -> RGrad()),
]

const GRID_MODELS = [
    ("exact",  () -> ExactHessian()),
    ("LBFGS",  () -> LBFGSModel(mem = 5)),
    ("SR1",    () -> SR1Model(mem = 5)),
    ("I",      () -> ScaledIdentity(c = 1.0)),
]

function interaction()
    arch = ExperimentArchive(tag = "interaction")
    save_config(arch; rules = GRID_RULES, params = SOLVER_PARAMS,
                problem_selection = PROBLEM_SELECTION,
                extra = Dict("experiment" => "exp6_interaction",
                             "models" => [m[1] for m in GRID_MODELS]))

    problems = default_problems()

    configs = [("$(rn)/$(mn)",
                () -> (rule = rf(), model = mf(), subsolver = DEFAULT_SUBSOLVER()))
               for (rn, rf) in GRID_RULES, (mn, mf) in GRID_MODELS]
    configs = vec(configs)

    records = run_experiment(problems, configs; params = SOLVER_PARAMS, archive = arch)

    # cell table: solved count and median iterations
    io = IOBuffer()
    @printf(io, "%-10s", "rule\\model")
    for (mn, _) in GRID_MODELS
        @printf(io, "%18s", mn)
    end
    println(io); println(io, "-"^(10 + 18length(GRID_MODELS)))
    solved_grid = zeros(length(GRID_RULES), length(GRID_MODELS))
    for (i, (rn, _)) in enumerate(GRID_RULES)
        @printf(io, "%-10s", rn)
        for (j, (mn, _)) in enumerate(GRID_MODELS)
            rs = filter(r -> r.config == "$(rn)/$(mn)", records)
            ok = filter(solved, rs)
            solved_grid[i, j] = length(ok) / max(length(problems), 1)
            med = isempty(ok) ? NaN : _median(Float64[r.iterations for r in ok])
            @printf(io, "%10d/%-3d%5s", length(ok), length(problems),
                    isnan(med) ? "--" : @sprintf("%.0f", med))
        end
        println(io)
    end
    println(io, "\n(cells: solved / total, then median iterations)")
    save_table(arch, "exp6_interaction_grid.txt", String(take!(io)))

    plt = heatmap([m[1] for m in GRID_MODELS], [r[1] for r in GRID_RULES],
                  solved_grid; xlabel = "model Hessian", ylabel = "radius rule",
                  clims = (0, 1), colorbar_title = "fraction solved")
    savefig_archived(arch, "exp6_interaction_heatmap.pdf", plt)

    # additivity check: does the grid factor into row + column effects?
    grand = sum(solved_grid) / length(solved_grid)
    rowe  = [sum(solved_grid[i, :]) / size(solved_grid, 2) for i in axes(solved_grid, 1)]
    cole  = [sum(solved_grid[:, j]) / size(solved_grid, 1) for j in axes(solved_grid, 2)]
    resid = [solved_grid[i, j] - rowe[i] - cole[j] + grand
             for i in axes(solved_grid, 1), j in axes(solved_grid, 2)]
    io = IOBuffer()
    println(io, "Additive-model residuals  (cell - row - column + grand mean)")
    println(io, "Large entries indicate a genuine rule×model interaction:")
    println(io, "the pair does something neither factor predicts alone.\n")
    @printf(io, "%-10s", "")
    for (mn, _) in GRID_MODELS; @printf(io, "%12s", mn); end
    println(io)
    for (i, (rn, _)) in enumerate(GRID_RULES)
        @printf(io, "%-10s", rn)
        for j in axes(resid, 2); @printf(io, "%12.4f", resid[i, j]); end
        println(io)
    end
    @printf(io, "\nmax |residual| = %.4f\n", maximum(abs, resid))
    save_table(arch, "exp6_additivity.txt", String(take!(io)))

    finalize_archive(arch; notes = """
        Full rule × model grid.

        exp6_additivity.txt reports how far the grid departs from a purely
        additive model. Small residuals support the survey's claim that the two
        axes act independently -- the rule sets step length, the model sets
        direction -- so their effects add rather than interact.

        The ScaledIdentity column is a control: that model carries no curvature
        at all, so any difference within it is attributable to the radius rule
        alone.
        """)
end

if abspath(PROGRAM_FILE) == @__FILE__
    interaction()
end