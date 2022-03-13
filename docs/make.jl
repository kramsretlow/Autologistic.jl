using Documenter
using Autologistic
using Graphs, DataFrames, CSV, Plots
makedocs(
    sitename = "Autologistic.jl",
    modules = [Autologistic],
    pages = [
        "index.md",
        "Background.md",
        "Design.md",
        "BasicUsage.md",
        "Examples.md",
        "api.md"
    ]
)
deploydocs(
    repo = "github.com/kramsretlow/Autologistic.jl.git",
)