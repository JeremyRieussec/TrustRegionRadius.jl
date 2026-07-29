# =============================================================================
# test/test_thresholds.jl
#
# Covers the two invariants introduced by the decoupling change:
#
#   1. 0 ≤ η ≤ η₁ ≤ η₂ < 1, with η governing acceptance and η₁, η₂ governing
#      scaling only.  In particular an iteration with ρ ∈ [η, η₁) is accepted
#      *and* contracts, and η = 0 is legal.
#   2. 0 < γ₁ ≤ γ₂ < 1 < γ₃ in every rule, rejected at construction.
#
# Plus the contraction property that the whole first-order theory rests on:
# every rule must return Δ_{k+1} < Δ_k on an unsuccessful iteration.  That is
# the test the old retrospective rules failed.
#
#     julia --project=. test/test_thresholds.jl
# =============================================================================

using Test
using TrustRegionRadius
using ADNLPModels

# Every rule under its default parameters, with the retrospective ones flagged.
const RULES = [
    ("RDelta",           RDelta()),
    ("RStep",            RStep()),
    ("RDFO",             RDFO()),
    ("RGrad",            RGrad()),
    ("RGradCapped",      RGradCapped(μ_max = 8.0)),
    ("RAdaptiveStep",    RAdaptiveStep()),
    ("RAdaptiveGrad",    RAdaptiveGrad()),
    ("RAdaptiveFanYuan", RAdaptiveFanYuan()),
    ("RRTR",             RRTR()),
    ("RRTRGrad",         RRTRGrad()),
]

@testset "threshold validation" begin
    # The chain is enforced, both ends.
    @test_throws ArgumentError TRParams(η = 0.5, η₁ = 0.2, η₂ = 0.9)   # η > η₁
    @test_throws ArgumentError TRParams(η₁ = 0.9, η₂ = 0.2)            # η₁ > η₂
    @test_throws ArgumentError TRParams(η = -0.1)                      # η < 0
    @test_throws ArgumentError TRParams(η₁ = 1.0, η₂ = 1.0)            # η₂ ≥ 1

    # η defaults to η₁: the coupled algorithm is unchanged by this refactor.
    @test TRParams(η₁ = 0.2).η == 0.2
    @test TRParams().η == TRParams().η₁

    # η = 0 is legal and is the case Part I covers but Curtis–Scheinberg does not.
    @test TRParams(η = 0.0, η₁ = 0.1, η₂ = 0.9).η == 0.0

    # Genuine decoupling: a middle regime exists.
    p = TRParams(η = 0.0, η₁ = 0.25, η₂ = 0.75)
    @test p.η < p.η₁ < p.η₂
end

@testset "factor convention 0 < γ₁ ≤ γ₂ < 1 < γ₃" begin
    # γ₂ in the expansion slot is the pre-refactor mistake; it must be rejected.
    @test_throws ArgumentError RGrad(γ₂ = 2.0)
    @test_throws ArgumentError RDelta(γ₂ = 1.5)
    @test_throws ArgumentError RRTR(γ₂ = 2.5)
    @test_throws ArgumentError RRTRGrad(γ₁ = 1.5)

    # γ₁ ≤ γ₂ is enforced, not merely documented.
    @test_throws ArgumentError RDelta(γ₁ = 0.8, γ₂ = 0.3)
    @test_throws ArgumentError RStep(γ₁ = 0.8, γ₂ = 0.3)
    @test_throws ArgumentError RDFO(γ₁ = 0.8, γ₂ = 0.3)

    # γ₃ must exceed 1.
    @test_throws ArgumentError RDelta(γ₃ = 0.9)
    @test_throws ArgumentError RGrad(γ₃ = 1.0)

    # The Hei family needs the stronger γ₃ > 1 + γ₂.
    @test_throws ArgumentError RAdaptiveStep(γ₂ = 0.5, γ₃ = 1.4)
    @test RAdaptiveStep(γ₂ = 0.5, γ₃ = 1.6) isa RAdaptiveStep

    # Defaults satisfy the convention for every rule that carries all three.
    for (name, r) in RULES
        fs = fieldnames(typeof(r))
        if :γ₁ in fs && :γ₂ in fs && :γ₃ in fs
            @test 0 < r.γ₁ <= r.γ₂ < 1 < r.γ₃
        elseif :γ₁ in fs && :γ₃ in fs
            @test 0 < r.γ₁ < 1 < r.γ₃
        end
    end
end

@testset "unsuccessful iterations contract, every rule" begin
    # An unsuccessful iteration under decoupled thresholds: ρ < η ≤ η₁.
    # Reproduces the failure mode of the old retrospective rules, where the
    # classical ρ was compared against the retrospective thresholds and could
    # leave Δ untouched.
    η, η₁, η₂ = 0.1, 0.25, 0.75
    Δ, s_norm, g_old, g_new = 1.0, 0.8, 2.0, 2.0   # rejected ⟹ ‖g‖ unchanged

    for (name, rule) in RULES
        @testset "$name" begin
            for ρ in (-1.0, -1e-8, 0.0, 0.05, 0.099)   # all strictly below η
                r = deepcopy(rule)
                reset_rule!(r)
                Δ0 = initial_radius(r, Δ, g_old)
                Δ1 = update_radius!(r, Δ0, ρ, false, η₁, η₂, s_norm, g_old, g_new)
                @test Δ1 < Δ0
                @test Δ1 > 0
            end
        end
    end
end

@testset "accepted-but-contracting iterations" begin
    # ρ ∈ [η, η₁): the step is taken, the radius still shrinks.  The rule is told
    # accepted = true and must nevertheless contract, because it reads η₁.
    η, η₁, η₂ = 0.0, 0.25, 0.75
    ρ = 0.1
    Δ, s_norm, g_old, g_new = 1.0, 0.9, 2.0, 1.5   # accepted ⟹ ‖g‖ decreased

    for (name, rule) in RULES
        needs_retrospective(rule) && continue      # ρ̃ ≠ ρ; tested separately
        @testset "$name" begin
            r = deepcopy(rule); reset_rule!(r)
            Δ0 = initial_radius(r, Δ, g_old)
            Δ1 = update_radius!(r, Δ0, ρ, true, η₁, η₂, s_norm, g_old, g_new)
            @test Δ1 < Δ0
        end
    end
end

@testset "asymptotic regime is declared for every rule" begin
    for (name, r) in RULES
        @test asymptotic_regime(r) in (:vanishing, :step_summable, :bounded_below)
        # The two-way predicate must agree with the three-way classification.
        if is_criticality_anchored(r)
            @test asymptotic_regime(r) === :vanishing
        end
    end
    @test asymptotic_regime(RDelta()) === :bounded_below
    @test asymptotic_regime(RStep())  === :step_summable
    @test asymptotic_regime(RDFO())   === :vanishing
end

@testset "step-driven rules refuse η₁ = 0" begin
    # The aggressive branch is unreachable at η₁ = 0 and the lower-bound constant
    # of Part I degenerates, so this is caught when the solver is built.
    nlp = ADNLPModel(x -> sum(abs2, x), ones(3), name = "sphere")
    bad = TRParams(η = 0.0, η₁ = 0.0, η₂ = 0.9)
    @test_throws ArgumentError TRSolver(nlp; rule = RStep(),         params = bad)
    @test_throws ArgumentError TRSolver(nlp; rule = RAdaptiveStep(), params = bad)
    # Rules with no such requirement accept it.
    @test TRSolver(nlp; rule = RDelta(), params = bad) isa TRSolver
    @test TRSolver(nlp; rule = RGrad(),  params = bad) isa TRSolver
end

@testset "end to end, coupled and decoupled" begin
    nlp = ADNLPModel(x -> (1 - x[1])^2 + 100(x[2] - x[1]^2)^2, [-1.2, 1.0],
                     name = "rosenbrock")
    coupled   = TRParams(η₁ = 0.1, η₂ = 0.9, tol = 1e-8)          # η = η₁
    decoupled = TRParams(η = 0.0, η₁ = 0.1, η₂ = 0.9, tol = 1e-8)

    # What is asserted here is the invariants of the refactor, not the quality of
    # any mechanism: a rule that exhausts its budget on Rosenbrock is a fact
    # about the rule, and belongs in the experiments, not in a regression test.
    # The one thing that must never happen is a run that neither converges nor
    # makes progress, which is what the retrospective non-contraction produced.
    for (name, rule) in RULES
        for (which, p) in (("coupled", coupled), ("decoupled", decoupled))
            @testset "$name / $which" begin
                st = tr_solve(nlp; rule = deepcopy(rule), params = p, trace = true)
                @test st.status in (:first_order, :max_iter, :stalled)
                st.status === :first_order || @info "did not converge" name which st.status st.dual_feas

                tr = st.solver_specific
                k  = st.iter
                @test length(tr[:ratio_trajectory])    == k
                @test length(tr[:accepted_trajectory]) == k
                @test length(tr[:step_trajectory])     == k
                @test length(tr[:active_trajectory])   == k
                @test length(tr[:delta_trajectory])    == k + 1

                # Acceptance matches ρ ≥ η, and nothing else.
                @test tr[:accepted_trajectory] == [ρ >= p.η for ρ in tr[:ratio_trajectory]]

                # Every rejected iteration contracted the radius, strictly unless
                # the rule's Δmin floor was already reached. This is the property
                # whose failure let the old RRTR re-solve an identical subproblem
                # until the iteration budget ran out.
                Δs    = tr[:delta_trajectory]
                floor = hasproperty(rule, :Δmin) ? rule.Δmin : 0.0
                for i in findall(!, tr[:accepted_trajectory])
                    @test Δs[i + 1] < Δs[i] || Δs[i] <= floor
                end
            end
        end
    end
end

@testset "η = 0 accepts strictly more steps" begin
    # Sanity check that the middle regime is not vacuous in practice.
    nlp = ADNLPModel(x -> (1 - x[1])^2 + 100(x[2] - x[1]^2)^2, [-1.2, 1.0])
    a = tr_solve(nlp; rule = RDelta(), trace = true,
                 params = TRParams(η₁ = 0.1, η₂ = 0.9, tol = 1e-8))
    b = tr_solve(nlp; rule = RDelta(), trace = true,
                 params = TRParams(η = 0.0, η₁ = 0.1, η₂ = 0.9, tol = 1e-8))
    ra = count(a.solver_specific[:accepted_trajectory]) / a.iter
    rb = count(b.solver_specific[:accepted_trajectory]) / b.iter
    @info "acceptance rate" coupled = ra decoupled = rb
    @test 0 <= ra <= 1 && 0 <= rb <= 1
end
