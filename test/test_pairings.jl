# Every model Hessian with every subproblem solver, each pairing run to convergence.

@testset "every model Hessian with every subsolver" begin
    # `ScaledIdentity` used to hand the subsolver `c * I`, which has no `size`, and
    # the Krylov.jl routines behind `KrylovCG` and `KrylovCR` read `size` before the
    # first product. Every pairing of that model with a Krylov subsolver raised a
    # `MethodError`, on any problem. The grid in test_solver.jl leaves the model out
    # and accepts `:exception` as a status, so here every pairing has to reach
    # `:first_order`.
    inner      = (SteihaugCG(), ExactMS(), KrylovCG(), KrylovCR())
    subsolvers = (inner..., map(EigenPoint, inner)...)

    # A linear regression on two coefficients is an NLSProblem, the narrowest problem
    # class, so every model is admissible on it and no pairing is skipped. A plain NLP
    # refuses the three outer-product models at construction, which
    # test_problem_classes.jl checks. The objective is a convex quadratic with Hessian
    # XᵀX/(Mσ²) and minimiser X \ y, which lies downhill from every other point, so
    # `SPDTarget` can be pointed at it.
    prob   = LinearRegression(K = 2, M = 50, seed = 1)
    xstar  = prob.X \ prob.y
    μ      = eigmin(Symmetric(prob.X' * prob.X ./ (prob.M * prob.σ^2)))
    models = (ExactHessian(), LBFGSModel(), SR1Model(), ScaledIdentity(),
              SPDTarget(target = xstar), BHHHModel(), BHHH2Model(), GaussNewtonModel())

    # Compared with the export list, so that a model or a subsolver added to the
    # package and not to these lists fails the suite.
    @test Set(nameof(typeof(m)) for m in models) == exported_concrete(ModelHessian)
    @test Set(nameof(typeof(s)) for s in subsolvers) == exported_concrete(SubproblemSolver)

    # `dense_max = 1` sends the outer-product models down their matrix-free branch,
    # which hands the subsolver an `OuterProductOperator` rather than a matrix.
    matrix_free = (BHHHModel(dense_max = 1), BHHH2Model(dense_max = 1),
                   GaussNewtonModel(dense_max = 1))

    # The Krylov wrappers stop on a residual of `atol + rtol‖g‖`, both `1e-6` by
    # default. A run whose ‖g‖ falls below about 1e-6 before reaching `tol` then gets
    # a zero step and ends `:stalled`. A tolerance above that holds every pairing to
    # `:first_order`.
    tol = 1e-5

    # One testset per pairing, so that a pairing which raises neither stops the
    # others nor goes unnamed in the report.
    for m in (models..., matrix_free...), sub in subsolvers
        @testset "$m with $sub" begin
            st = tr_solve(FullBatchNLP(prob); rule = RDelta(), model = m,
                          subsolver = sub,
                          params = TRParams(tol = tol, max_iterations = 1_000))
            @test st.status === :first_order
            # g = H(x − x*), so ‖g‖ ≤ tol puts x within tol/λ_min(H) of x*.
            @test norm(st.solution .- xstar) <= tol / μ
        end
    end

    # The case as reported, on a two-variable ADNLPModel. The objective is
    # 2-strongly convex with minimiser (1, −1/2), so ‖x − x*‖ ≤ ‖g‖/2.
    bowl(x) = (x[1] - 1)^2 + 4 * (x[2] + 0.5)^2 + (x[1] - 1)^4
    for sub in subsolvers
        @testset "ScaledIdentity on an ADNLPModel with $sub" begin
            st = tr_solve(ADNLPModel(bowl, [-1.2, 1.0]); rule = RDelta(),
                          model = ScaledIdentity(), subsolver = sub,
                          params = TRParams(tol = tol, max_iterations = 1_000))
            @test st.status === :first_order
            @test norm(st.solution .- [1.0, -0.5]) <= tol / 2
        end
    end
end
