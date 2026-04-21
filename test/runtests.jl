using Test
using LinearAlgebra
using NLPModels
using SolverCore
using ADNLPModels
using TrustRegionRadius

# ------------------------------------------------------------
# Test problems
# ------------------------------------------------------------

rosenbrock(x) = 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2

function make_rosenbrock(T::Type = Float64)
    x0 = T[-1.2, 1.0]
    return ADNLPModel(rosenbrock, x0)
end

quadratic(x) = sum(abs2, x)

function make_quadratic(n::Int = 10, T::Type = Float64)
    x0 = ones(T, n)
    return ADNLPModel(quadratic, x0)
end

# ============================================================
# 1. Constructor tests
# ============================================================

@testset "TRRSolver constructor" begin
    nlp = make_quadratic(5)

    solver = TRRSolver(nlp)
    @test solver isa TRRSolver
    @test length(solver.x)      == 5
    @test length(solver.g)      == 5
    @test length(solver.p)      == 5
    @test length(solver.x_cand) == 5
    @test length(solver.Hp)     == 5
    @test solver.rule      isa R1ClassicalUpdate
    @test solver.subsolver isa SteihaugTointCG

    # Custom rule + subsolver
    solver2 = TRRSolver(nlp;
        rule      = R4RelativeGradUpdate(),
        subsolver = SteihaugTointCG(; max_iters = 50),
    )
    @test solver2.rule      isa R4RelativeGradUpdate
    @test solver2.subsolver.max_iters == 50
end

# ============================================================
# 2. Return type and convergence on a simple problem
# ============================================================

@testset "solve! returns GenericExecutionStats" begin
    nlp = make_quadratic(5)
    stats = trust_region_radius(nlp; params = TRSolverParams(tol = 1e-8))
    @test stats isa GenericExecutionStats
    @test stats.status == :first_order
    @test stats.dual_feas <= 1e-8
end

@testset "convergence on Rosenbrock -- all canonical rules" begin
    params = TRSolverParams(tol = 1e-6, max_iterations = 2_000)
    for rule in (R1ClassicalUpdate(),
                 R2StepSizeUpdate(),
                 R3DFOLikeUpdate(),
                 R4RelativeGradUpdate())
        nlp   = make_rosenbrock()
        stats = trust_region_radius(nlp; rule=rule, params=params)
        @test stats.status == :first_order
        @test norm(stats.solution .- [1.0, 1.0]) < 1e-3
    end
end

@testset "convergence on Rosenbrock -- Hei family" begin
    params = TRSolverParams(tol = 1e-6, max_iterations = 2_000)
    hei_default = (0.25, 0.1, 0.25, 2.0, 4.0, 2.0, 2.0)   # η, β, γ₁, γ₂, M, λ₁, λ₂
    for rule in (HeiUpdate(hei_default...),
                 HeiGradUpdate(hei_default...),
                 HeiFanYuanUpdate(1.0, hei_default...))
        nlp   = make_rosenbrock()
        stats = trust_region_radius(nlp; rule=rule, params=params)
        @test stats.status == :first_order
    end
end

# ============================================================
# 3. NLPModels counters are populated
# ============================================================

@testset "NLPModels counters are used" begin
    nlp = make_rosenbrock()
    stats = trust_region_radius(nlp; params = TRSolverParams(tol = 1e-6))
    @test neval_obj(nlp)   > 0
    @test neval_grad(nlp)  > 0
    @test neval_hprod(nlp) > 0
end

# ============================================================
# 4. reset_rule! / SolverCore.reset!
# ============================================================

@testset "reset_rule! restores mutable state" begin
    rule = R4RelativeGradUpdate(0.25, 2.0, 1.0)
    μ0 = rule.μ
    # mutate
    rule.μ = 17.0
    reset_rule!(rule)
    @test rule.μ == μ0

    fy = HeiFanYuanUpdate(1.0, 0.25, 0.1, 0.25, 2.0, 4.0, 2.0, 2.0)
    μ0_fy = fy.μ
    fy.μ = 42.0
    reset_rule!(fy)
    @test fy.μ == μ0_fy
end

@testset "Solver.reset! is callable" begin
    nlp = make_quadratic(3)
    solver = TRRSolver(nlp; rule = R4RelativeGradUpdate())
    solver.rule.μ = 99.0
    SolverCore.reset!(solver)
    @test solver.rule.μ == solver.rule.μ₀
end

@testset "rule state is reset at the start of solve!" begin
    nlp = make_quadratic(5)
    rule = R4RelativeGradUpdate()
    μ0 = rule.μ
    stats1 = trust_region_radius(nlp; rule=rule)
    # Caller's rule was deep-copied, so caller's μ is untouched:
    @test rule.μ == μ0
    # A second run starts fresh:
    stats2 = trust_region_radius(nlp; rule=rule)
    @test stats2.status == :first_order
end

# ============================================================
# 5. Callback mechanism
# ============================================================

@testset "callback is called every iteration and can stop the run" begin
    nlp = make_rosenbrock()
    counter = Ref(0)
    function cb(nlp, solver, stats)
        counter[] += 1
        if stats.iter >= 3
            stats.status = :user
        end
    end
    stats = trust_region_radius(nlp;
        params   = TRSolverParams(tol = 1e-12, max_iterations = 100),
        callback = cb)
    @test counter[] >= 3
    @test stats.status == :user
end

# ============================================================
# 6. Pluggable subsolver -- all three concrete implementations
# ============================================================

@testset "all subsolvers reach first-order on quadratic" begin
    params = TRSolverParams(tol = 1e-8, max_iterations = 200)
    for sub in (SteihaugTointCG(), KrylovCG(), KrylovCGLanczos())
        nlp   = make_quadratic(20)
        stats = trust_region_radius(nlp; subsolver=sub, params=params)
        @test stats.status == :first_order
    end
end

# ============================================================
# 7. Float32 parametric support
# ============================================================

@testset "works with Float32 NLPModel" begin
    nlp = make_quadratic(5, Float32)
    params = TRSolverParams{Float32}(tol = 1f-4, max_iterations = 200)
    stats = trust_region_radius(nlp; params = params)
    @test stats.status == :first_order
    @test eltype(stats.solution) == Float32
end

# ============================================================
# 8. In-place / functional interfaces agree
# ============================================================

@testset "in-place solve! and functional wrapper agree" begin
    nlp1 = make_rosenbrock()
    nlp2 = make_rosenbrock()
    params = TRSolverParams(tol = 1e-8, max_iterations = 500)

    # Functional
    s_func = trust_region_radius(nlp1; params=params)

    # In-place
    solver = TRRSolver(nlp2; params=params)
    stats  = GenericExecutionStats(nlp2)
    s_ip   = solve!(solver, nlp2, stats)

    @test s_func.status == s_ip.status
    @test norm(s_func.solution .- s_ip.solution) < 1e-6
end
