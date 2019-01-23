using Documenter, Autologistic
makedocs(
    sitename = "Autologistic.jl",
    modules = [Autologistic],
    pages = [
        "index.md",
        "api.md"
    ]
)
deploydocs(
    repo = "github.com/kramsretlow/Autologistic.jl.git",
)