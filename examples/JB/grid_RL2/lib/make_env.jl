# https://pymdp-rtd.readthedocs.io/en/latest/notebooks/clue_chaining_demo.html
# Active Inference Demo: Epistemic Chaining

# include("./grid.jl")
#show(stdout, "text/plain", x)
# @infiltrate; @assert false

import Random
import Distributions

include("./utils.jl")
include("./structs.jl")


####################################################################################################

mutable struct RLEnv
    init_loc::NamedTuple{<:Any, <:NTuple{N, Int64} where {N}}
    model::NamedTuple
    nX::Int
    nY::Int
    current_loc::NamedTuple{<:Any, <:NTuple{N, Int64} where {N}}
    adjacencies_dic::Dict
    opposite_dic::Dict
    moves_dic::Dict
end

function RLEnv(
    model,
    start_cell 
    )
    
    nY, nX = maximum(first.(model.states.loc.labels)), maximum(last.(model.states.loc.labels))
    current_loc = ((loc_obs=findfirst(x -> x == start_cell, model.states.loc.labels)),)
    
    #@infiltrate; @assert false
    return RLEnv(
        current_loc, 
        model,
        nX,
        nY,
        current_loc, 
        Dict(),  # adjacencies_dic
        
        # opposites_dic, for building walls
        Dict(
            :UP => :DOWN,
            :DOWN => :UP,
            :LEFT => :RIGHT,
            :RIGHT => :LEFT
        ),

        # moves_dic
        Dict(
            :UP => [-1, 0], # UP
            :DOWN => [1,0],  # DOWN
            :RIGHT => [0, 1], # RIGHT
            :LEFT => [0,-1], # LEFT
            :STAY => [0, 0]  # STAY
        ),

    )
end


function add_walls(env::RLEnv, walls)
    
    # add adjacencies dict
    for i in 1:env.nY
        env.adjacencies_dic[i] = Dict()
        for j in 1:env.nX
            env.adjacencies_dic[i][j] = collect(keys(env.moves_dic))
        end
    end
    
    for walls_ in walls
        for (cell, edge) in walls_
            # remove action 
            filter!(x -> x != edge, env.adjacencies_dic[cell[1]][cell[2]])
            
            # remove opposite action
            new_cell = cell + env.moves_dic[edge]
            filter!(x -> x != env.opposite_dic[edge], env.adjacencies_dic[new_cell[1]][new_cell[2]])
        end
    end

    #@infiltrate; @assert false
end


function step_env!(env::RLEnv, action::Union{Nothing, NamedTuple{<:Any, <:NTuple{N, Int64} where {N}}})
    # use nothing as a null action if null actions are not otherwise specified in model
    
    loc = env.current_loc  # e.g., (loc_obs=7,)
    loc_id = values(loc)[1]

    cell = env.model.states.loc.labels[loc_id]
    action_ids = values(action)

    if length(env.model.actions) == 1
        policy = env.model.actions.move.labels[action_ids[1]]  # e.g., :LEFT,
    else
        
        if action_ids == (1, 3)
            policy = :UP
        elseif action_ids == (2, 3)
            policy = :DOWN
        elseif action_ids == (3, 3)
            policy = :STAY
        elseif action_ids == (3, 1)
            policy = :LEFT
        elseif action_ids == (3, 2)
            policy = :RIGHT
        else
            @infiltrate; @assert false
        end
    end

    if !(policy in env.adjacencies_dic[cell[1]][cell[2]])
        return loc
    end
    
    new_cell = cell + env.moves_dic[policy] 
    if any(new_cell .< [1,1]) || any(new_cell .> [env.nY, env.nX])
        return loc
    else
        new_loc_id = findfirst(x -> x == new_cell, env.model.states.loc.labels)
        env.current_loc = (loc_obs=new_loc_id,)
        return (loc_obs=new_loc_id,)
    end

    #@infiltrate; @assert false
end


function reset_env!(env::RLEnv)
    # Reset environment to initial location
    env.current_loc = env.init_loc
    #println("Re-initialized location to $(env.init_loc)")
    return env.current_loc
end




