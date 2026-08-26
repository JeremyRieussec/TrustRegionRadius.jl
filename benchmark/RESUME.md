# Reprendre une campagne interrompue

## En une ligne

```bash
TRR_RESUME=benchmark/results/exp_2026-07-29_02-15-23_comparison \
  julia --project=benchmark benchmark/experiments/exp1_comparison.jl
```
TRR_RESUME=benchmark/results/exp_2026-08-22_20-59-14_convergence_rate \ 
julia --project=benchmark benchmark/experiments/exp7_convergence_rate.jl 

This syntax is a feature of Unix shells (Bash, Zsh, etc.). It temporarily creates an environment variable only for the command that follows.

Before starting Julia, the shell creates
```bash
TRR_RESUME = "benchmark/results/exp_2026-07-29_02-15-23_comparison"
```
This variable exists only in the environment of the process that is about to be launched.

It then executes
```bash
julia --project=benchmark benchmark/experiments/exp1_comparison
```

and passes its environment to the new Julia process.

So Julia starts with
```bash
ENV = Dict(
    ...
    "PATH" => "...",
    "HOME" => "...",
    "TRR_RESUME" => "benchmark/results/exp_2026-07-29_02-15-23_comparison",
    ...
)
```
So, inside Julia,
```julia
ENV["TRR_RESUME"] == "benchmark/results/exp_2026-07-29_02-15-23_comparison"
```  
And the constructor `ExperimentArchive(tag = ...)` will reopen the indicated archive instead of creating a new one, and `run_experiment` will skip any execution whose `.jld2` file already exists. For this, the script `benchmark/experiments/exp1_comparison.jl` contains the line
```julia 
dir_resume = get(ENV, "TRR_RESUME", "")
```

Le script d'expérience n'a **pas** besoin d'être modifié. La variable
d'environnement fait que `ExperimentArchive(tag = ...)` rouvre l'archive
indiquée au lieu d'en créer une nouvelle, et `run_experiment` saute toute
exécution dont le fichier `.jld2` existe déjà.

Sous PowerShell :

```powershell
$env:TRR_RESUME = "benchmark\results\exp_2026-07-29_02-15-23_comparison"
julia --project=benchmark benchmark\experiments\exp1_comparison.jl
Remove-Item Env:TRR_RESUME     # à ne pas oublier
```

## Retrouver l'archive interrompue

```julia
include("benchmark/archive.jl")
latest_archive()                       # la plus récente
readdir(joinpath(latest_archive(), "data"))   # ce qui a déjà été calculé
```

## Ce qui se passe à la reprise

Pour chaque problème, `run_experiment` regarde d'abord si **toutes** ses
configurations sont en cache. Si oui, le modèle n'est pas ouvert du tout : sur
CUTEst, l'ouverture décode et compile un fichier SIF, et c'est ce coût qui
domine une reprise. Sinon le modèle est ouvert et seules les configurations
manquantes sont recalculées.

Les exécutions relues sont marquées d'une astérisque dans le tableau de
progression, et un décompte est affiché à la fin :

```
problem                     RDelta         RStep          RDFO         RGrad
--------------------------------------------------------------------------
ROSENBR                        27*           31*            29            34
WOOD                           45*           52*            48            51

4 exécution(s) relue(s) du cache (marquées *), 4 recalculée(s).
```

Un fichier `.jld2` tronqué par l'interruption est illisible : il est alors
traité comme absent, avec un avertissement, et l'exécution est refaite. Il n'y
a donc rien à nettoyer à la main.

## Refaire seulement les figures

Si toutes les exécutions sont là et que seule la mise en forme doit changer :

```julia
include("benchmark/archive.jl")
include("benchmark/harness.jl")

arch    = reopen_archive(latest_archive())
records = load_records(arch)          # aucun solveur n'est relancé
```

`records` est un `Vector{RunRecord}` identique à celui qu'aurait produit la
campagne ; il alimente directement `metric_matrix`, `success_table` et les
tracés.

## Désactiver la reprise

```julia
run_experiment(problems, configs; params = SOLVER_PARAMS,
               archive = arch, resume = false)
```

Tout est alors recalculé et les fichiers existants sont écrasés.

---

# Trois corrections apportées au passage

## 1. Les noms de fichiers contenant « / » faisaient échouer l'expérience 6

Les configurations de `exp6_interaction.jl` s'appellent `"RDelta/exact"`.
L'ancien code écrivait `data/$(problème)_$(config).jld2`, donc
`data/ROSENBR_RDelta/exact.jld2` : un sous-répertoire inexistant, et `jldsave`
lève une exception.

Comme cet appel se trouvait **en dehors** du bloc `try`, l'exception remontait
et arrêtait toute la campagne. C'est un candidat sérieux pour expliquer l'arrêt
observé.

Désormais `data_filename` assainit `/ \ : * ? " < > |`, et l'écriture est
protégée par son propre `try` : une campagne de plusieurs heures ne peut plus
être perdue parce qu'un nom de fichier a déplu au système.

## 2. Le nom de fichier a changé

L'ancien format était `PROBLÈME_CONFIG.jld2`, le nouveau
`PROBLÈME__CONFIG.jld2` (deux tirets bas). Le séparateur simple était ambigu
dès qu'un nom de problème contenait lui-même un tiret bas.

**Conséquence pour vous :** les fichiers déjà écrits ne seront pas reconnus. Une
commande de renommage :

```julia
d = joinpath(latest_archive(), "data")
for f in readdir(d)
    endswith(f, ".jld2") || continue
    occursin("__", f) && continue
    dict = JLD2.load(joinpath(d, f))
    new  = string(replace(dict["problem_name"], r"[/\\:*?\"<>|]" => "-"), "__",
                  replace(dict["rule_name"],    r"[/\\:*?\"<>|]" => "-"), ".jld2")
    mv(joinpath(d, f), joinpath(d, new); force = true)
end
```

Sinon, laisser la campagne recalculer : c'est plus lent mais sans risque.

## 3. `h_evals` n'était pas enregistré

`RunRecord` porte `h_evals` (le compteur de produits hessienne-vecteur), mais
il ne figurait pas parmi les champs écrits. Une reprise aurait donc perdu cette
colonne. Il est maintenant sauvegardé, et `_record_from_data` le met à zéro
s'il est absent, pour rester compatible avec les fichiers déjà produits.
