# Model Hessians: the operator contract and the SPDTarget existence condition.

@testset "model Hessians" begin
    nlp = ADNLPModel(x -> x[1]^2 + 2x[2]^2, [1.0, 1.0])
    x = [1.0, 1.0]
    v = [1.0, 0.0]

    @testset "operator contract" begin
        # `size` and `eltype` as well as `*` and `mul!`, because the Krylov.jl routines
        # behind KrylovCG and KrylovCR read both before the first product. This loop
        # checked `*` alone, and ScaledIdentity passed it while returning `c * I`,
        # which has no size, so every Krylov pairing of that model raised.
        lr = FullBatchNLP(LinearRegression(K = 2, M = 50, seed = 1))
        cases = ((ExactHessian(), nlp), (LBFGSModel(mem = 3), nlp),
                 (SR1Model(mem = 3), nlp), (ScaledIdentity(c = 2.0), nlp),
                 (SPDTarget(target = [0.0, 0.0]), nlp),
                 # the outer-product models need scores, and `dense_max = 1`
                 # selects their matrix-free branch
                 (BHHHModel(), lr), (BHHH2Model(), lr), (GaussNewtonModel(), lr),
                 (BHHHModel(dense_max = 1), lr), (BHHH2Model(dense_max = 1), lr),
                 (GaussNewtonModel(dense_max = 1), lr))
        @test Set(nameof(typeof(m)) for (m, _) in cases) == exported_concrete(ModelHessian)
        for (m, p) in cases
            @testset "$m" begin
                reset_model!(m, 2)
                B = hessian_op(m, p, x)
                @test size(B) == (2, 2)
                @test eltype(B) === Float64
                Bv = B * v
                @test length(Bv) == 2
                @test all(isfinite, Bv)
                y = fill(NaN, 2)
                mul!(y, B, v)
                @test y ≈ Bv
            end
        end
        # The scaled identity takes the element type of the problem.
        @test eltype(hessian_op(ScaledIdentity(), nlp, Float32[1, 1])) === Float32
    end

    @testset "ExactHessian is exact" begin
        m = ExactHessian()
        @test hessian_op(m, nlp, x) * v ≈ [2.0, 0.0]
        @test dense_hessian(m, nlp, x) ≈ [2.0 0.0; 0.0 4.0]
    end

    @testset "ScaledIdentity carries no curvature" begin
        m = ScaledIdentity(c = 3.0)
        @test (hessian_op(m, nlp, x) * v) ≈ 3.0 * v
    end

    @testset "quasi-Newton models absorb secant pairs" begin
        for m in (LBFGSModel(mem = 3), SR1Model(mem = 3))
            reset_model!(m, 2)
            s = [0.1, 0.0]; y = [0.2, 0.0]
            @test update_model!(m, s, y) === nothing
            @test all(isfinite, hessian_op(m, nlp, x) * v)
        end
    end

    @testset "SPDTarget: exists iff the target lies downhill" begin
        # φ(x) = g(x)ᵀ(target − x) < 0 is necessary for ANY SPD model whose
        # minimiser is the target; a DomainError is the honest report.
        m = SPDTarget(target = [0.0, 0.0])
        φ = phi_target(m, nlp, x)
        @test φ < 0                                  # origin is downhill from (1,1)
        B = dense_hessian(m, nlp, x)
        @test isposdef(Symmetric(B))
        # the model minimiser is the target: −B⁻¹g == target − x
        @test -(Symmetric(B) \ grad(nlp, x)) ≈ [0.0, 0.0] .- x  atol=1e-8

        uphill = SPDTarget(target = [5.0, 5.0])
        @test phi_target(uphill, nlp, x) > 0
        @test_throws DomainError dense_hessian(uphill, nlp, x)
    end
end
