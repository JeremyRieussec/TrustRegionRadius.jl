using Documenter
using TrustRegionRadius

DocMeta.setdocmeta!(TrustRegionRadius, :DocTestSetup,
                    :(using TrustRegionRadius); recursive = true)

makedocs(
    sitename = "TrustRegionRadius.jl",
    modules  = [TrustRegionRadius],
    authors  = "Jérémy Rieussec, Fabian Bastin",
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical  = "https://JeremyRieussec.github.io/TrustRegionRadius.jl",
        assets     = String[],
    ),
    pages = [
        "Home"          => "index.md",
        "Getting started" => "quickstart.md",
        "The three axes"  => [
            "Radius mechanisms"   => "rules.md",
            "Model Hessians"      => "models.md",
            "Subproblem solvers"  => "subsolvers.md",
        ],
        "Benchmarking"  => "benchmarking.md",
        "API reference" => "api.md",
    ],
    # A missing docstring is a real defect in a package whose point is to be read
    # alongside a paper; fail the build rather than warn.
    checkdocs = :exports,
    warnonly  = false,
)

deploydocs(
     repo      = "github.com/JeremyRieussec/TrustRegionRadius.jl.git",
     devbranch = "main"
)
