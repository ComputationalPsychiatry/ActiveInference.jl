# @infiltrate; @assert false

################################################################################
################################ ENVIRONMENT ###################################
################################################################################

mutable struct SI_MAZE_ENV
    maze::Matrix{Int}
    agent_row::Int
    agent_col::Int
    model
    history::Vector{NamedTuple}
end


function init_env(model, START)
    start_id = findfirst(x -> x == START, model.states.loc.labels)
    nrows = size(model.states.loc.extra.MAZE, 1)
    row = (start_id - 1) % nrows + 1
    col = Int((start_id - 1) ÷ nrows + 1)
    env = SI_MAZE_ENV(model.states.loc.extra.MAZE, row, col, model, NamedTuple[])
    initial_obs = get_observation(env)
    push!(env.history, initial_obs)
    return env
end


function step_env!(env::SI_MAZE_ENV, action::Union{Nothing, NamedTuple{<:Any, <:NTuple{N, Int64} where {N}}})
    ACTIONS = Dict(
        1 => (-1, 0),  # UP
        2 => ( 1, 0),  # DOWN
        3 => ( 0, -1), # LEFT
        4 => ( 0, 1),  # RIGHT
        5 => ( 0, 0)   # STAY
    )

    if isnothing(action)
        # null action
        obs = get_observation(env)
        push!(env.history, obs)
        return obs
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

    obs = get_observation(env)
    push!(env.history, obs)
    
    #@infiltrate; @assert false
    return obs
end


function get_observation(env::SI_MAZE_ENV)
    r, c = env.agent_row, env.agent_col

    # 1st modality: position index 
    pos_index = (c - 1) * size(env.maze, 1) + r

    # 2nd modality: 
    maze_obs = env.maze[r, c] + 1

    # 3rd modality: valence (probabilistic)
    p_safe = env.model.obs.valence_obs.extra.PREFERENCES_matrix[pos_index]  # safe probability
    valence = rand() < p_safe ? 1 : 2

    return (; zip([x.name for x in env.model.obs], [pos_index, maze_obs, valence])...)
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
