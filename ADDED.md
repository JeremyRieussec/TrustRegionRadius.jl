# Files added in this pass

Drop these into the package root, keeping the paths.

```
TrustRegionRadius/
├── Project.toml              REPLACED  — real UUID, compat bounds, test target
├── README.md                 REPLACED  — badges, tutorial link
├── LICENSE                   NEW  — MIT; change if your institution requires otherwise
├── CITATION.cff              NEW  — software + preferred citation to Part I
├── SETUP.md                  NEW  — the remaining steps, in order
├── ADDED.md                  NEW  — this file
│
├── .github/workflows/
│   ├── CI.yml                NEW  — tests on LTS + stable × 3 OSes, plus a parse check
│   │                                for the benchmark scripts
│   ├── Documenter.yml        NEW  — builds and publishes docs to gh-pages
│   ├── CompatHelper.yml      NEW  — keeps [compat] current in all three environments
│   └── TagBot.yml            NEW  — releases on registration
│
├── benchmark/Project.toml    REPLACED  — real UUID
│
├── docs/
│   ├── Project.toml          REPLACED  — real UUID, Documenter compat
│   ├── make.jl               REPLACED  — six pages, checkdocs = :exports, deploydocs
│   └── src/
│       ├── index.md          REPLACED
│       ├── quickstart.md     NEW
│       ├── rules.md          NEW
│       ├── models.md         NEW
│       ├── subsolvers.md     NEW
│       ├── benchmarking.md   NEW
│       └── api.md            NEW
│
└── notebooks/
    └── tutorial.ipynb        NEW  — 33 cells, eight sections
```

## Substitutions to make

Replace `USER` with your GitHub username in:

- `README.md` (four badge URLs and two links)
- `CITATION.cff` (`repository-code`)
- `docs/make.jl` (`canonical` and `deploydocs(repo = ...)`)
- `docs/src/index.md`, `quickstart.md` (install instructions)
- `notebooks/tutorial.ipynb` (final section)

One pass does it:

```bash
grep -rl 'USER' . --include='*.md' --include='*.jl' --include='*.cff' --include='*.ipynb' \
  | xargs sed -i 's/USER/your-github-username/g'
```

## The UUID

`7a8d5bce-8bd3-4f8e-8a33-850f3a7a2263`

Already written into all three `Project.toml` files. Generate your own with
`using UUIDs; uuid4()` if you prefer — but change all three together.

## Documentation coverage

All 47 exported symbols appear in exactly one `@docs` block across the six pages; verified
mechanically. `docs/make.jl` sets `checkdocs = :exports` and `warnonly = false`, so a missing
or malformed docstring fails the build rather than passing silently — appropriate for a
package meant to be read alongside a paper.

## Not done

- **`Manifest.toml` files** are deliberately absent and git-ignored. A package pins `[compat]`
  ranges, not exact versions.
- **Codecov** needs a `CODECOV_TOKEN` secret for private repositories; the step is marked
  `continue-on-error` so CI stays green without it.
- **Zenodo** archiving is described in `SETUP.md` §10 but needs a release to trigger.
