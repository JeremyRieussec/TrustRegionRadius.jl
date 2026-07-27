using Documenter
using TrustRegionRadius

makedocs(
    sitename = "TrustRegionRadius.jl",
    modules  = [TrustRegionRadius],
    format   = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages    = ["Home" => "index.md"],
)

# deploydocs(repo = "github.com/…/TrustRegionRadius.jl.git")

# docs/make.jl has deploydocs commented out, since it needs your actual repository URL.