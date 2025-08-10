# @infiltrate; @assert false

################################################################################
################################ ENVIRONMENT ###################################
################################################################################

mutable struct SI_MAZE_ENV{T<:AbstractFloat}
    maze::Matrix{Int}
    preferences::Matrix{T}
    agent_row::Int
    agent_col::Int
    model
end


function init_env(model, START, float_type::Type)
    start_id = findfirst(x -> x == START, model.states.loc.labels)
    nrows = size(model.states.loc.extra.MAZE, 1)
    row = (start_id - 1) % nrows + 1
    col = Int((start_id - 1) ÷ nrows + 1)
    PREF = float_type.(model.obs.safe_obs.extra.PREFERENCES)
    return SI_MAZE_ENV(model.states.loc.extra.MAZE, PREF, row, col, model)
end


const ACTIONS = Dict(
    1 => (-1, 0),  # UP
    2 => ( 0, 1),  # RIGHT
)


function step_env!(env::SI_MAZE_ENV, action::Union{Nothing, NamedTuple{<:Any, <:NTuple{N, Int64} where {N}}})
    
    if isnothing(action)
        # null action
        return get_observation(env)
    end

    dr, dc = ACTIONS[action[1]]

    r′ = env.agent_row + dr
    c′ = env.agent_col + dc

    if 1 ≤ r′ ≤ size(env.maze, 1) &&
       1 ≤ c′ ≤ size(env.maze, 2) &&
       env.maze[r′, c′] == 0
        env.agent_row = r′
        env.agent_col = c′
    end

    #@infiltrate; @assert false
    return get_observation(env)
end


function get_observation(env::SI_MAZE_ENV)
    r, c = env.agent_row, env.agent_col
    pos_index = (c - 1) * size(env.maze, 1) + r
    valence = rand() < env.preferences[r, c] ? 1 : 2 
    
    #@infiltrate; @assert false
    return (; zip([x.name for x in env.model.obs], [pos_index, 1, valence])...)
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
