# Model Hessians: the operator contract and the SPDTarget existence condition.

@testset "model Hessians" begin
    nlp = ADNLPModel(x -> x[1]^2 + 2x[2]^2, [1.0, 1.0])
    x = [1.0, 1.0]
    v = [1.0, 0.0]

    @testset "operator contract" begin
        for m in (ExactHessian(), LBFGSModel(mem = 3), SR1Model(mem = 3),
                  ScaledIdentity(c = 2.0))
            reset_model!(m, 2)
            B = hessian_op(m, nlp, x)
            @test length(B * v) == 2
            @test all(isfinite, B * v)
        end
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
