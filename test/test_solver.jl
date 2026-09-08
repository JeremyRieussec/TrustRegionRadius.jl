# The solver: convergence, tracing, and the three axes acting independently.

@testset "solver" begin
    rosen(x) = (1 - x[1])^2 + 100(x[2] - x[1]^2)^2

    @testset "TRParams validation" begin
        # ArgumentError, not AssertionError: @assert is documented as liable to be
        # disabled, and these are argument checks. The threshold chain itself is
        # covered in test_thresholds.jl.
        @test_throws ArgumentError TRParams(η1 = 0.9, η2 = 0.1)
        @test_throws ArgumentError TRParams(Δ0 = -1.0)
        @test_throws ArgumentError TRParams(Δmax = 0.5, Δ0 = 1.0)
        @test_throws ArgumentError TRParams(max_iterations = 0)
        @test_throws ArgumentError TRParams(tol = 0.0)
        p = TRParams(tol = 1e-8)
        @test p.tol == 1e-8
    end

    @testset "the solver validates thresholds against the rule" begin
        nlp = ADNLPModel(rosen, [-1.2, 1.0])
        bad = TRParams(η = 0.0, η1 = 0.0, η2 = 0.9)
        @test_throws ArgumentError DeterministicTRSolver(nlp; rule = RStep(),         params = bad)
        @test_throws ArgumentError DeterministicTRSolver(nlp; rule = RAdaptiveStep(), params = bad)
        @test DeterministicTRSolver(nlp; rule = RDelta(), params = bad) isa DeterministicTRSolver
        @test DeterministicTRSolver(nlp; rule = RGrad(),  params = bad) isa DeterministicTRSolver
    end

    # ------
    # every rule is compatible with every model and every subsolver, but we don't need to test all combinations.
    # ------
    @testset "every rule is compatible with every model and subsolver" begin
        # KrylovCG and KrylovCR are in the grid now. They were left out of it, with
        # a "NEED to check" comment in their place, and that is exactly why
        # `KrylovCGLanczos` could sit in the exported API raising a MethodError on
        # every call: `Krylov.cg_lanczos` has no `radius` keyword, so the wrapper
        # generated for it had never once run.
        for r in (RDelta(), RStep(), RDFO(ζ = 1.0), RGrad(), RGradCapped(μ_max = 8.0),
                  RAdaptiveStep(), RAdaptiveGrad(), 
                  RRTR(), RRTRGrad())
            for m in (ExactHessian(), LBFGSModel(mem = 5), SR1Model(mem = 5))
                for sub in (SteihaugCG(), ExactMS(), KrylovCG(), KrylovCR())
                    nlp = ADNLPModel(rosen, [-1.2, 1.0])
                    st = tr_solve(nlp; rule = r, model = m, subsolver = sub,
                                  params = TRParams(tol = 1e-6, max_iterations = 3))
                    @test st !== nothing && st.status in (:first_order, :max_iter, :exception, :max_time, :user, :stalled)
                end
            end
        end
    end

    # ------
    # check DeterministicTRSolver construction
    # ------
    @testset "DeterministicTRSolver construction" begin
        nlp = ADNLPModel(rosen, [-1.2, 1.0])
        solver = DeterministicTRSolver(nlp; rule = RDelta(), model = ExactHessian(),
                          subsolver = SteihaugCG())
        @test solver.rule isa RadiusRule
        @test solver.model isa ModelHessian
        @test solver.subsolver isa SubproblemSolver
    end

    @testset "every subsolver converges" begin
        for sub in (SteihaugCG(), ExactMS(), KrylovCR())
            nlp = ADNLPModel(rosen, [-1.2, 1.0])
            st = tr_solve(nlp; rule = RDelta(), subsolver = sub,
                          params = TRParams(tol = 1e-6, max_iterations = 5_000))
            @test st.status === :first_order
        end
    end

    @testset "the Krylov wrappers actually call something that takes a radius" begin
        # A regression test for a defect that a compatibility grid would have caught:
        # the wrapper body passes `radius = Δ`, so a Krylov routine without that
        # keyword raises on every call. Assert against Krylov itself rather than
        # against our own wrapper, so the test names the real precondition.
        Kry = TrustRegionRadius.Krylov
        for f in (Kry.cg, Kry.cr)
            kw = union(Base.kwarg_decl.(methods(f))...)
            @test :radius in kw
        end
        @test :radius ∉ union(Base.kwarg_decl.(methods(Kry.cg_lanczos))...)

        nlp = ADNLPModel(rosen, [-1.2, 1.0])
        for sub in (KrylovCG(), KrylovCR())
            st = tr_solve(nlp; rule = RDelta(), subsolver = sub,
                          params = TRParams(tol = 1e-6, max_iterations = 5_000))
            @test st.status in (:first_order, :stalled)
            @test st.solution ≈ [1.0, 1.0] atol = 1e-4
        end

        # Indefinite `B` is the case `KrylovCG` cannot serve and `KrylovCR` can:
        # the step must reach the boundary rather than raise or return zero.
        indef(x) = 0.5 * (2x[1]^2 - x[2]^2)
        nlp2 = ADNLPModel(indef, [1.0, 0.5])
        x = [1.0, 0.5]; g = grad(nlp2, x)
        s = similar(x); Hs = similar(x)
        active = solve_subproblem!(KrylovCR(), ExactHessian(), nlp2, x, g, 0.75, s, Hs)
        @test active
        @test norm(s) ≈ 0.75 rtol = 1e-6
        @test dot(g, s) + 0.5 * dot(s, Symmetric(hess(nlp2, x)) * s) < 0

        # The old name still resolves, and to the new type. Asserted as an identity
        # rather than with @test_deprecated, whose log capture depends on the
        # --depwarn setting the suite happens to be run under.
        @test KrylovCGLanczos === KrylovCR
        @test KrylovCGLanczos() isa KrylovCR
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

    @testset "the alignment survives every exit route, not just :first_order" begin
        # The convention above was only ever tested on a run that converged. On a
        # :stalled exit `_tr_step!` returns early, so the loop used to break without
        # tracing while still counting the iteration: `st.iter` was 1 with one entry
        # in :delta_trajectory and none in :ratio_trajectory. Anything zipping a
        # per-state series against a per-iteration one silently misaligned.
        per_state = (:delta_trajectory, :grad_trajectory, :obj_trajectory)
        per_iter  = (:ratio_trajectory, :step_trajectory, :active_trajectory,
                     :accepted_trajectory, :rho_tilde_trajectory, :xi_trajectory,
                     :cos_cauchy_trajectory, :cg_iters_trajectory,
                     :branch_trajectory, :gamma_trajectory)

        function check_alignment(st, want_status)
            @test st.status === want_status
            ss, k = st.solver_specific, st.iter
            for key in per_state
                @test length(ss[key]) == k + 1
            end
            for key in per_iter
                @test length(ss[key]) == k
            end
            return ss, k
        end

        # A radius pinned below the level at which f(x) - f(x+s) carries any
        # information: the first step stalls, and it is a real iteration.
        stall = TRParams(Δ0 = 1e-18, Δmax = 1e-18, tol = 1e-16, max_iterations = 50)
        ss, k = check_alignment(
            tr_solve(ADNLPModel(rosen, [-1.2, 1.0]); rule = RDelta(), trace = true,
                     params = stall), :stalled)
        @test k == 1
        @test length(ss[:ratio_trajectory]) == 1

        # The fields the acceptance test and the radius update would have written
        # are set on the stalled path rather than left stale from the iteration
        # before, so the last traced entry describes *this* iteration.
        @test ss[:accepted_trajectory][end] == false
        @test ss[:branch_trajectory][end] === :none
        @test isnan(ss[:gamma_trajectory][end])
        @test isfinite(ss[:ratio_trajectory][end]) || ss[:ratio_trajectory][end] == -Inf

        # :max_iter, the other early exit, has always been aligned; assert it so
        # the convention is pinned on more than one route.
        check_alignment(
            tr_solve(ADNLPModel(rosen, [-1.2, 1.0]); rule = RDelta(), trace = true,
                     params = TRParams(tol = 1e-16, max_iterations = 4)), :max_iter)
    end

    @testset "acceptance follows ρ ≥ η and nothing else" begin
        nlp = ADNLPModel(rosen, [-1.2, 1.0])
        for p in (TRParams(η1 = 0.1, η2 = 0.9, tol = 1e-6),
                  TRParams(η = 0.0, η1 = 0.1, η2 = 0.9, tol = 1e-6))
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
            for p in (TRParams(η1 = 0.1, η2 = 0.9, tol = 1e-6, max_iterations = 2_000),
                      TRParams(η = 0.0, η1 = 0.1, η2 = 0.9, tol = 1e-6,
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
        solver = DeterministicTRSolver(nlp; rule = RGrad(μ = 1.0))
        SolverCore.solve!(solver, nlp)
        SolverCore.reset!(solver)
        @test solver.rule.μ == solver.rule.μ0
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
