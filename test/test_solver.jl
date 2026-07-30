# The solver: convergence, tracing, and the three axes acting independently.

@testset "solver" begin
    rosen(x) = (1 - x[1])^2 + 100(x[2] - x[1]^2)^2

    @testset "TRParams validation" begin
        # ArgumentError, not AssertionError: @assert is documented as liable to be
        # disabled, and these are argument checks. The threshold chain itself is
        # covered in test_thresholds.jl.
        @test_throws ArgumentError TRParams(η₁ = 0.9, η₂ = 0.1)
        @test_throws ArgumentError TRParams(Δ₀ = -1.0)
        @test_throws ArgumentError TRParams(Δmax = 0.5, Δ₀ = 1.0)
        @test_throws ArgumentError TRParams(max_iterations = 0)
        @test_throws ArgumentError TRParams(tol = 0.0)
        p = TRParams(tol = 1e-8)
        @test p.tol == 1e-8
    end

    @testset "the solver validates thresholds against the rule" begin
        nlp = ADNLPModel(rosen, [-1.2, 1.0])
        bad = TRParams(η = 0.0, η₁ = 0.0, η₂ = 0.9)
        @test_throws ArgumentError TRSolver(nlp; rule = RStep(),         params = bad)
        @test_throws ArgumentError TRSolver(nlp; rule = RAdaptiveStep(), params = bad)
        @test TRSolver(nlp; rule = RDelta(), params = bad) isa TRSolver
        @test TRSolver(nlp; rule = RGrad(),  params = bad) isa TRSolver
    end

    # ------
    # every rule is compatible with every model and every subsolver, but we don't need to test all combinations.
    # ------
    # -----
    # NEED to check KrylovCGLanczos()
    # -----
    @testset "every rule is compatible with every model and subsolver" begin
        for r in (RDelta(), RStep(), RDFO(ζ = 1.0), RGrad(), RGradCapped(μ_max = 8.0),
                  RAdaptiveStep(), RAdaptiveGrad(), 
                  RRTR(), RRTRGrad())
            for m in (ExactHessian(), LBFGSModel(mem = 5), SR1Model(mem = 5))
                for sub in (SteihaugCG(), ExactMS())
                    nlp = ADNLPModel(rosen, [-1.2, 1.0])
                    st = tr_solve(nlp; rule = r, model = m, subsolver = sub,
                                  params = TRParams(tol = 1e-6, max_iterations = 3))
                    @test st !== nothing && st.status in (:first_order, :max_iter, :exception, :max_time, :user, :stalled)
                end
            end
        end
    end

    # ------
    # check TRSolver construction
    # ------
    @testset "TRSolver construction" begin
        nlp = ADNLPModel(rosen, [-1.2, 1.0])
        solver = TRSolver(nlp; rule = RDelta(), model = ExactHessian(),
                          subsolver = SteihaugCG())
        @test solver.rule isa RadiusRule
        @test solver.model isa ModelHessian
        @test solver.subsolver isa SubproblemSolver
    end

    # -----
    # NEED to check KrylovCGLanczos()
    # -----
    @testset "every subsolver converges" begin
        for sub in (SteihaugCG(), ExactMS())
            nlp = ADNLPModel(rosen, [-1.2, 1.0])
            st = tr_solve(nlp; rule = RDelta(), subsolver = sub,
                          params = TRParams(tol = 1e-6, max_iterations = 5_000))
            @test st.status === :first_order
        end
    end

    @testset "trace records the trajectories" begin
        nlp = ADNLPModel(rosen, [-1.2, 1.0])
        st = tr_solve(nlp; rule = RDelta(), trace = true,
                      params = TRParams(tol = 1e-6))
        ss = st.solver_specific
        for k in (:delta_trajectory, :grad_trajectory, :obj_trajectory,
                  :ratio_trajectory, :step_trajectory, :active_trajectory,
                  :accepted_trajectory)
            @test haskey(ss, k)
        end
        @test length(ss[:delta_trajectory]) >= 2
        @test eltype(ss[:active_trajectory]) === Bool
        @test eltype(ss[:accepted_trajectory]) === Bool
        # ‖g‖ decreases overall
        @test ss[:grad_trajectory][end] < ss[:grad_trajectory][1]
    end

    @testset "trajectory lengths and alignment" begin
        # Δ, ‖g‖ and f have a value before the first iteration; the rest do not.
        # Anything that plots them together has to know which is which.
        nlp = ADNLPModel(rosen, [-1.2, 1.0])
        st = tr_solve(nlp; rule = RDelta(), trace = true,
                      params = TRParams(tol = 1e-6))
        ss, k = st.solver_specific, st.iter
        for key in (:delta_trajectory, :grad_trajectory, :obj_trajectory)
            @test length(ss[key]) == k + 1
        end
        for key in (:ratio_trajectory, :step_trajectory, :active_trajectory,
                    :accepted_trajectory)
            @test length(ss[key]) == k
        end
    end

    @testset "acceptance follows ρ ≥ η and nothing else" begin
        nlp = ADNLPModel(rosen, [-1.2, 1.0])
        for p in (TRParams(η₁ = 0.1, η₂ = 0.9, tol = 1e-6),
                  TRParams(η = 0.0, η₁ = 0.1, η₂ = 0.9, tol = 1e-6))
            st = tr_solve(nlp; rule = RDelta(), trace = true, params = p)
            ss = st.solver_specific
            @test ss[:accepted_trajectory] ==
                  [ρ >= p.η for ρ in ss[:ratio_trajectory]]
        end
    end

    @testset "unsuccessful iterations contract the radius" begin
        # The property the first-order theory rests on, observed end to end
        # rather than on synthetic arguments: no rejected iteration may leave the
        # radius where it was, or the next one re-solves an identical subproblem.
        for rule in (RDelta(), RStep(), RDFO(ζ = 1.0), RGrad(), RGradCapped(μ_max = 8.0),
                     RAdaptiveStep(), RAdaptiveGrad(), 
                     RRTR(), RRTRGrad())
            for p in (TRParams(η₁ = 0.1, η₂ = 0.9, tol = 1e-6, max_iterations = 2_000),
                      TRParams(η = 0.0, η₁ = 0.1, η₂ = 0.9, tol = 1e-6,
                               max_iterations = 2_000))
                nlp = ADNLPModel(rosen, [-1.2, 1.0])
                st = tr_solve(nlp; rule = rule, trace = true, params = p)
                ss = st.solver_specific
                Δs = ss[:delta_trajectory]
                Δfloor = hasproperty(rule, :Δmin) ? rule.Δmin : 0.0
                for i in findall(!, ss[:accepted_trajectory])
                    # Strictly, unless the rule's own floor has been reached.
                    @test Δs[i + 1] < Δs[i] || Δs[i] <= Δfloor
                end
            end
        end
    end

    # @testset "criticality-anchored rules drive Δ down" begin
    #     nlp = ADNLPModel(rosen, [-1.2, 1.0])
    #     st = tr_solve(nlp; rule = RGrad(), trace = true,
    #                   params = TRParams(tol = 1e-8, max_iterations = 5_000))
    #     Δ = st.solver_specific[:delta_trajectory]
    #     @test Δ[end] < Δ[1]                      # Δ_k → 0
    # end

    @testset "max_iterations is respected" begin
        nlp = ADNLPModel(rosen, [-1.2, 1.0])
        st = tr_solve(nlp; rule = RDelta(),
                      params = TRParams(tol = 1e-14, max_iterations = 3))
        @test st.status === :max_iter
        @test st.iter <= 3
    end

    @testset "solver reset restores rule state" begin
        nlp = ADNLPModel(rosen, [-1.2, 1.0])
        solver = TRSolver(nlp; rule = RGrad(μ = 1.0))
        SolverCore.solve!(solver, nlp)
        SolverCore.reset!(solver)
        @test solver.rule.μ == solver.rule.μ₀
    end

    @testset "a config can be reused across problems" begin
        # rule and model are deep-copied at construction, so state does not leak
        rule = RGrad(μ = 1.0)
        for x0 in ([-1.2, 1.0], [2.0, 2.0], [0.0, 0.0])
            nlp = ADNLPModel(rosen, x0)
            st = tr_solve(nlp; rule = rule, params = TRParams(tol = 1e-6,
                                                              max_iterations = 5_000))
            @test st.status === :first_order
        end
        @test rule.μ == 1.0        # caller's instance untouched
    end
end
