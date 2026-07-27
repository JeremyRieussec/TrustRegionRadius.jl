# =============================================================================
# src/Benchmark/profiles.jl
#
# Performance profiles (Dolan & Moré 2002) and data profiles
# (Moré & Wild 2009), plus a pgfplots exporter.
#
# Both take a cost matrix `T[p, s]` with `Inf` (or `NaN`) marking failure, and
# both are pure functions of that matrix: nothing here touches a solver, so the
# same code serves a run of this package or externally produced numbers.
# =============================================================================

"""
    performance_profile(T; ntau = 500, taumax = nothing) -> (τ, prof)

Performance profile of Dolan & Moré (2002).

`T[p, s]` is the cost of solver `s` on problem `p`; use `Inf` or `NaN` for a
failure. Returns the abscissa `τ` and the matrix

    prof[i, s] = fraction of problems with r_{p,s} ≤ τ[i],
    r_{p,s} = T[p,s] / min_{s'} T[p,s'].

`prof[1, s]` is the fraction of problems on which `s` is fastest (efficiency);
the right-hand asymptote is the fraction it solves at all (reliability).

!!! note "Ranking depends on the solver set"
    Ratios are normalised by the per-problem best, so adding or removing a
    solver can reorder the curves (Gould & Scott 2016). Report pairwise
    profiles against a fixed baseline alongside the full comparison.

# Example
```julia
τ, prof = performance_profile(T)
open(io -> profile_to_pgfplots(io, τ, prof, labels), "prof.tex", "w")
```
"""
function performance_profile(T::AbstractMatrix; ntau::Int = 500, taumax = nothing)
    npb, ns = size(T)
    npb > 0 && ns > 0 || throw(ArgumentError("performance_profile: empty cost matrix"))

    Tc = [(isfinite(T[i, j]) && T[i, j] > 0) ? float(T[i, j]) : Inf
          for i in 1:npb, j in 1:ns]
    best = [minimum(@view Tc[i, :]) for i in 1:npb]

    R = [(isfinite(Tc[i, j]) && isfinite(best[i])) ? Tc[i, j] / best[i] : Inf
         for i in 1:npb, j in 1:ns]

    fin = filter(isfinite, vec(R))
    rmax = isempty(fin) ? 2.0 : maximum(fin)
    tmax = taumax === nothing ? max(1.05 * rmax, 2.0) : taumax

    τ = exp10.(range(0.0, log10(tmax), length = ntau))
    # NOTE: named `prof`, not `π`, to avoid shadowing `Base.π` for callers
    # who `using TrustRegionRadius` and then write `2π`.
    prof = [count(<=(t), @view R[:, j]) / npb for t in τ, j in 1:ns]
    return τ, prof
end

"""
    data_profile(N, dims; nkappa = 500, kmax = nothing) -> (κ, d)

Data profile of Moré & Wild (2009).

`N[p, s]` is the number of function evaluations solver `s` needed on problem
`p` (`Inf` if unsolved) and `dims[p]` the dimension of problem `p`. The
abscissa is the budget measured in units of `n+1` evaluations — one simplex
gradient — which is what makes problems of different size comparable.

Unlike the performance profile, the data profile is not normalised by the best
solver, so it does not suffer the set-dependence noted above; it answers
"what fraction is solved within this budget?" directly.
"""
function data_profile(N::AbstractMatrix, dims::AbstractVector;
                      nkappa::Int = 500, kmax = nothing)
    npb, ns = size(N)
    length(dims) == npb || throw(DimensionMismatch(
        "data_profile: length(dims) = $(length(dims)) but N has $npb rows"))

    U = [isfinite(N[i, j]) ? N[i, j] / (dims[i] + 1) : Inf for i in 1:npb, j in 1:ns]
    fin = filter(isfinite, vec(U))
    km = kmax === nothing ? (isempty(fin) ? 10.0 : maximum(fin)) : kmax

    κ = collect(range(0.0, km, length = nkappa))
    d = [count(<=(k), @view U[:, j]) / npb for k in κ, j in 1:ns]
    return κ, d
end

"""
    profile_to_pgfplots(io, x, y, labels; npoints = 60, styles = nothing, logx = true)

Write `\\addplot` blocks ready to `\\input` into a `pgfplots` axis.

The curves are subsampled onto a logarithmic grid (or a linear one if
`logx = false`, appropriate for data profiles) so the emitted file stays small:
a profile computed on a 2000-point grid becomes ~60 coordinates per curve with
no visible difference.

`styles`, if given, is a vector of pgfplots option strings, one per curve.

# Example
```julia
τ, prof = performance_profile(T)
open("prof_mech.tex", "w") do io
    profile_to_pgfplots(io, τ, prof, ["R-delta", "R-step", "R-DFO", "R-grad"];
                        styles = ["black,solid", "black,dashed",
                                  "gray,solid", "gray,dashdotted"])
end
```
then in LaTeX:
```latex
\\begin{axis}[xmode=log, xlabel={\$\\tau\$}, ylabel={\$\\pi_s(\\tau)\$}]
  \\input{prof_mech.tex}
\\end{axis}
```
"""
function profile_to_pgfplots(io::IO, x::AbstractVector, y::AbstractMatrix,
                             labels::AbstractVector;
                             npoints::Int = 60,
                             styles = nothing,
                             logx::Bool = true)
    nx = length(x)
    size(y, 1) == nx || throw(DimensionMismatch(
        "profile_to_pgfplots: y has $(size(y,1)) rows but x has $nx entries"))
    size(y, 2) == length(labels) || throw(DimensionMismatch(
        "profile_to_pgfplots: y has $(size(y,2)) columns but $(length(labels)) labels"))

    idx = if logx
        unique(round.(Int, exp10.(range(0, log10(nx), length = npoints))))
    else
        unique(round.(Int, range(1, nx, length = npoints)))
    end
    idx = filter(i -> 1 <= i <= nx, idx)

    for (j, lab) in enumerate(labels)
        pts = join((@sprintf("(%.5g,%.4f)", x[i], y[i, j]) for i in idx), " ")
        opt = styles === nothing ? "" : "[" * styles[j] * "]"
        println(io, "% ", lab)
        println(io, "\\addplot", opt, " coordinates {", pts, "};")
        println(io, "\\addlegendentry{", lab, "}")
    end
    return nothing
end

profile_to_pgfplots(x, y, labels; kwargs...) =
    profile_to_pgfplots(stdout, x, y, labels; kwargs...)
