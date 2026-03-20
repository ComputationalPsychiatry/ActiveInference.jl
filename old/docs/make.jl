using ActiveInference
using Documenter
using Literate

# Set project directory
if haskey(ENV, "GITHUB_WORKSPACE")
    project_dir = ENV["GITHUB_WORKSPACE"]
    input_folder = joinpath(project_dir, "docs", "julia_files")
else
    project_dir = pwd()
    input_folder = raw"..\julia_files"
end

cd(joinpath(project_dir, "docs", "src"))

DocMeta.setdocmeta!(ActiveInference, :DocTestSetup, :(using ActiveInference); recursive=true)

# Automating the creating of the markdown files
julia_files = filter(file -> endswith(file, ".jl"), readdir(input_folder))

for file in julia_files
    input_path = joinpath(input_folder, file)
    Literate.markdown(input_path, outputdir="", execute=true, documenter=true, codefence =  "```julia" => "```")
end

# Creating the documentation
makedocs(;
    modules=[ActiveInference, ActiveInference.Environments],
    authors="Jonathan Ehrenreich Laursen, Samuel William Nehrer",
    repo="https://github.com/ilabcode/ActiveInference.jl/blob/{commit}{path}#{line}",
    sitename="ActiveInference.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ilabcode.github.io/ActiveInference.jl",
        edit_link="master",
        assets=[],
    ),
    pages=[

        "General Introduction" => [

            "Introduction" => "Introduction.md",
            "Creation of the Generative Model" => "GenerativeModelCreation.md",
            "Creating the Agent" => "AgentCreation.md",
            "Simulation" => "Simulation.md",
            "Model Fitting" => "Fitting.md",
            "Simulation with ActionModels.jl" => "SimulationActionModels.md",

        ],

        "Usage Examples" => [

            "T-Maze Simulation" => "TMazeSimulationExample.md",
            # "T-Maze Model Fitting" => [],

        ],

        "Theory" => [

            # "Active Inference Theory" => [
            #     "Perception" => [],
            #     "Action" => [],
            #     "Learning" => [],
            # ],

            "POMDP Theory" => "GenerativeModelTheory.md",



        ],

        # "Why Active Inference?" => "WhyActiveInference.md",

        "Index" => "index.md",
    ],
    doctest=true,
)

deploydocs(;
    repo="github.com/ilabcode/ActiveInference.jl",
    devbranch="master",
)
