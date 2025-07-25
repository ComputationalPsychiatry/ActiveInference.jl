MAZE  = [
    1 1 1 1 1 1 1 1 1;
    1 1 1 1 1 1 1 0 1;
    1 1 1 1 1 0 1 0 1;
    1 1 1 0 1 0 1 0 1;
    1 0 1 0 1 0 1 0 1;
    1 0 1 0 1 0 1 0 1;
    1 0 1 0 1 0 1 0 1;
    1 0 0 0 0 0 0 0 1;
    1 0 1 1 1 1 1 1 1
]

LOCATIONS = [
    1  10  19  28  37  46  55  64  73;
    2  11  20  29  38  47  56  65  74;
    3  12  21  30  39  48  57  66  75;
    4  13  22  31  40  49  58  67  76;
    5  14  23  32  41  50  59  68  77;
    6  15  24  33  42  51  60  69  78;
    7  16  25  34  43  52  61  70  79;
    8  17  26  35  44  53  62  71  80;
    9  18  27  36  45  54  63  72  81
]

PREFERENCES = [
    0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000;
    0.000  0.000  0.000  0.000  0.000  0.000  0.000  1.000  0.000;
    0.000  0.000  0.000  0.000  0.000  0.700  0.000  0.825  0.000;
    0.000  0.000  0.000  0.500  0.000  0.600  0.000  0.700  0.000;
    0.000  0.300  0.000  0.400  0.000  0.525  0.000  0.625  0.000;
    0.000  0.200  0.000  0.300  0.000  0.425  0.000  0.575  0.000;
    0.000  0.150  0.000  0.200  0.000  0.325  0.000  0.500  0.000;
    0.000  0.100  0.125  0.150  0.175  0.225  0.300  0.400  0.000;
    0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000
]
################################################################################
################################ ENVIRONMENT ###################################
################################################################################

mutable struct SI_MAZE_ENV
    maze::Matrix{Int}
    preferences::Matrix{Float64}
    agent_row::Int
    agent_col::Int
end

function init_env(MAZE::Matrix{Int}, PREFERENCES::Matrix{Float64}, START::Int)
    nrows = size(MAZE, 1)
    row = (START - 1) % nrows + 1
    col = Int((START - 1) ÷ nrows + 1)
    return SI_MAZE_ENV(MAZE, PREFERENCES, row, col)
end

const ACTIONS = Dict(
    1 => (-1, 0),  # UP
    2 => ( 0, 1),  # RIGHT
)

function step!(env::SI_MAZE_ENV, action::Vector{<:Real})
    a = Int(action[1])
    dr, dc = ACTIONS[a]

    r′ = env.agent_row + dr
    c′ = env.agent_col + dc

    if 1 ≤ r′ ≤ size(env.maze, 1) &&
       1 ≤ c′ ≤ size(env.maze, 2) &&
       env.maze[r′, c′] == 0
        env.agent_row = r′
        env.agent_col = c′
    end

    return get_observation(env)
end

function get_observation(env::SI_MAZE_ENV)::Vector{Int}
    r, c = env.agent_row, env.agent_col
    pos_index = (c - 1) * size(env.maze, 1) + r
    valence = rand() < env.preferences[r, c] ? 1 : 2 
    return [pos_index, 1, valence]
end

#=
using Plots

cgrad([:red, :salmon], 10)

heatmap(PREFERENCES, color=cgrad([:red4, :lightsalmon], 256), aspect_ratio=:equal, xlabel="Columns", ylabel="Rows",
    title="Preferences", c=:viridis, xticks=1:9, yticks=1:9,
    xlims=(0.5, 9.5), ylims=(0.5, 9.5), yflip=true,
    grid=false, framestyle=:box, legend=false, clims=(0,1),
    colorbar_title="Preference Value", colorbar_ticks=[0, 0.25, 0.5, 0.75, 1.0],
    colorbar_tickvals=[0, 0.25, 0.5, 0.75, 1.0],
    colorbar_ticklabels=["0", "0.25", "0.5", "0.75", "1.0"],
    colorbar_label="Preference Value")

# Add location indices as white text
for i in 1:size(LOCATIONS, 1)
    for j in 1:size(LOCATIONS, 2)
        annotate!(j, i, text(string(LOCATIONS[i, j]), :white, :center, 7))
    end
end

heatmap!(replace(MAZE, 0 => missing), color=cgrad(:greys, rev=true), aspect_ratio=:equal, xlabel="Columns", ylabel="Rows",
    title="Horizon = 9", c=:greys, xticks=1:9, yticks=1:9,
    xlims=(0.5, 9.5), ylims=(0.5, 9.5), yflip=true,
    grid=false, framestyle=:box, legend=false, clims=(0,1))

=#
