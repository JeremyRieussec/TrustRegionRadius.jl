# Profiles: the definitions of Dolan–Moré and Moré–Wild, on hand-checkable data.

@testset "profiles" begin
    @testset "performance_profile basics" begin
        # solver 1 wins on problem 1, solver 2 on problem 2, solver 3 fails once
        T = [1.0 2.0 4.0;
             2.0 1.0 Inf]
        τ, prof = performance_profile(T)
        @test size(prof, 2) == 3
        @test prof[1, 1] ≈ 0.5        # fastest on 1 of 2
        @test prof[1, 2] ≈ 0.5
        @test prof[1, 3] ≈ 0.0
        @test prof[end, 1] ≈ 1.0      # reliability
        @test prof[end, 3] ≈ 0.5      # solved only one problem
        @test all(diff(prof[:, 1]) .>= -1e-12)   # non-decreasing in τ
    end

    @testset "performance_profile: failures are Inf" begin
        T = [1.0 Inf; 1.0 Inf]
        _, prof = performance_profile(T)
        @test prof[end, 2] == 0.0
    end

    @testset "data_profile scales by dimension" begin
        N = [10.0 20.0; 100.0 50.0]
        dims = [9, 49]                      # budgets of 1 and 2 simplex gradients
        κ, d = data_profile(N, dims)
        @test κ[1] == 0.0
        @test all(0 .<= d .<= 1)
        @test d[end, 1] ≈ 1.0
    end

    @testset "data_profile checks dims length" begin
        @test_throws DimensionMismatch data_profile([1.0 2.0], [1, 2, 3])
    end

    @testset "pgfplots export" begin
        T = [1.0 2.0; 2.0 1.0]
        τ, prof = performance_profile(T)
        io = IOBuffer()
        profile_to_pgfplots(io, τ, prof, ["a", "b"])
        out = String(take!(io))
        @test occursin("\\addplot", out)
        @test occursin("\\addlegendentry{a}", out)
        @test count("\\addplot", out) == 2
    end

    @testset "pgfplots label count is checked" begin
        T = [1.0 2.0; 2.0 1.0]
        τ, prof = performance_profile(T)
        @test_throws DimensionMismatch profile_to_pgfplots(IOBuffer(), τ, prof, ["only one"])
    end
end
