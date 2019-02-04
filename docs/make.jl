# TODO: figure out how to properly handle external dependencies.
using Documenter, Autologistic, LightGraphs, DataFrames, CSV
makedocs(
    sitename = "Autologistic.jl",
    modules = [Autologistic],
    pages = [
        "index.md",
        "Background.md",
        "Examples.md",
        "api.md"
    ]
)
deploydocs(
    repo = "github.com/kramsretlow/Autologistic.jl.git",
)