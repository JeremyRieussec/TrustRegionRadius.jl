# The solver: convergence, tracing, and the three axes acting independently.

@testset "solver" begin
    rosen(x) = (1 - x[1])^2 + 100(x[2] - x[1]^2)^2

    @testset "TRParams validation" begin
        @test_throws AssertionError TRParams(η₁ = 0.9, η₂ = 0.1)
        @test_throws AssertionError TRParams(Δ₀ = -1.0)
        p = TRParams(tol = 1e-8)
        @test p.tol == 1e-8
    end

    @testset "every rule converges on Rosenbrock" begin
        for r in (RDelta(), RStep(), RDFO(ζ = 1.0), RGrad(),
                  RAdaptiveStep(), RAdaptiveGrad(), RRTR(), RRTRGrad())
            nlp = ADNLPModel(rosen, [-1.2, 1.0])
            st = tr_solve(nlp; rule = r, params = TRParams(tol = 1e-6,
                                                            max_iterations = 5_000))
            @test st.status === :first_order
            @test st.solution ≈ [1.0, 1.0] atol=1e-3
        end
    end

    @testset "every model converges" begin
        for m in (ExactHessian(), LBFGSModel(mem = 5), SR1Model(mem = 5))
            nlp = ADNLPModel(rosen, [-1.2, 1.0])
            st = tr_solve(nlp; rule = RDelta(), model = m,
                          params = TRParams(tol = 1e-6, max_iterations = 5_000))
            @test st.status === :first_order
        end
    end

    @testset "every subsolver converges" begin
        for sub in (SteihaugCG(), ExactMS(), KrylovCGLanczos())
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
                  :ratio_trajectory, :step_trajectory, :active_trajectory)
            @test haskey(ss, k)
        end
        @test length(ss[:delta_trajectory]) >= 2
        @test eltype(ss[:active_trajectory]) === Bool
        # ‖g‖ decreases overall
        @test ss[:grad_trajectory][end] < ss[:grad_trajectory][1]
    end

    @testset "criticality-anchored rules drive Δ down" begin
        nlp = ADNLPModel(rosen, [-1.2, 1.0])
        st = tr_solve(nlp; rule = RGrad(), trace = true,
                      params = TRParams(tol = 1e-8, max_iterations = 5_000))
        Δ = st.solver_specific[:delta_trajectory]
        @test Δ[end] < Δ[1]                      # Δ_k → 0
    end

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
        solve!(solver, nlp)
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
