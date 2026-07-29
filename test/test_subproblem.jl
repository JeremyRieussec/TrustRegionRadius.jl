# Subproblem solvers: optimality, boundary detection, and the hard case.

@testset "subproblem solvers" begin
    nlp = ADNLPModel(x -> 0.5x[1]^2 + x[2]^2, [1.0, 1.0])
    x = [1.0, 1.0]
    g = grad(nlp, x)
    s = similar(g); H_buf = similar(g)

    @testset "interior step is the Newton step" begin
        for sub in (SteihaugCG(), ExactMS())
            active = solve_subproblem!(sub, ExactHessian(), nlp, x, g, 100.0, s, H_buf)
            @test !active
            @test s ≈ -(Symmetric(dense_hessian(ExactHessian(), nlp, x)) \ g) atol=1e-6
        end
    end

    @testset "small radius: step is on the boundary" begin
        for sub in (SteihaugCG(), ExactMS())
            active = solve_subproblem!(sub, ExactHessian(), nlp, x, g, 0.01, s, H_buf)
            @test active
            @test norm(s) ≈ 0.01 atol=1e-8
        end
    end

    @testset "first CG iterate is the Cauchy point" begin
        # This is the diagnostic behind the μ_max threshold: when the region is
        # small enough that CG truncates immediately, the model Hessian has had
        # no influence on the direction at all.
        info = cg_step_info(SteihaugCG(), ExactHessian(), nlp, x, g, 1e-6)
        @test info.cg_iters == 1
        @test info.active
        @test info.cos_cauchy ≈ 1.0 atol=1e-12
    end

    @testset "large radius: CG takes more than one iteration" begin
        info = cg_step_info(SteihaugCG(), ExactHessian(), nlp, x, g, 100.0)
        @test info.cg_iters >= 1
        @test !info.active
    end

    @testset "ExactMS handles indefinite B" begin
        # A saddle: the Newton step is not a descent direction, so the solver
        # must follow negative curvature to the boundary.
        ind = ADNLPModel(y -> 0.5y[1]^2 - 0.5y[2]^2, [1.0, 1.0])
        xi = [1.0, 1.0]; gi = grad(ind, xi)
        si = similar(gi); Hi = similar(gi)
        active = solve_subproblem!(ExactMS(), ExactHessian(), ind, xi, gi, 0.5, si, Hi)
        @test active
        @test norm(si) ≈ 0.5 atol=1e-8
    end

    @testset "ExactMS refuses oversized problems" begin
        big = ADNLPModel(y -> sum(abs2, y), ones(300))
        xb = ones(300); gb = grad(big, xb)
        sb = similar(gb); Hb = similar(gb)
        @test_throws ArgumentError solve_subproblem!(ExactMS(nmax = 200),
            ExactHessian(), big, xb, gb, 1.0, sb, Hb)
    end

    @testset "zero gradient gives a zero step" begin
        z = ADNLPModel(y -> sum(abs2, y), [0.0, 0.0])
        xz = [0.0, 0.0]; gz = grad(z, xz)
        sz = similar(gz); Hz = similar(gz)
        @test !solve_subproblem!(SteihaugCG(), ExactHessian(), z, xz, gz, 1.0, sz, Hz)
        @test norm(sz) == 0
    end
end
