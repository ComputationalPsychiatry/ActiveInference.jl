
# @infiltrate; @assert false



####################################################################################################



function make_B(model, CONFIG)
    ### B matrix
        
    B = model.states.loc.B

    # complete uncertainty about transitions
    B = B .+ 1. ./prod(model.states.loc.extra.grid_dims)

    model = @set model.states.loc.B = B
    #@infiltrate; @assert false

    return model
end



function make_B_true(model, env, CONFIG)
    # no uncertainty in loc
    B_true = zeros(model.states.loc.B_dims)

    for loc_id in 1:B_true.size[2]
        cell = model.states.loc.labels[loc_id] # e.g., (9,3)
        for action in model.policies.action_iterator
            if length(model.actions) == 1
                policy = model.actions.move.labels[action[1]]  # e.g., :LEFT,
            else
                #@infiltrate; @assert false
                if action == (1, 3)
                    policy = :UP
                elseif action == (2, 3)
                    policy = :DOWN
                elseif action == (3, 3)
                    policy = :STAY
                elseif action == (3, 1)
                    policy = :LEFT
                elseif action == (3, 2)
                    policy = :RIGHT
                else
                    @infiltrate; @assert false
                end
            end
            
            if !(policy in env.adjacencies_dic[cell[1]][cell[2]])
                new_cell = cell
            else
                new_cell = cell + env.moves_dic[policy] 
                if any(new_cell .< [1,1]) || any(new_cell .> [env.nY, env.nX])
                    new_cell = cell
                end
            end
            new_cell_id = findfirst(x -> x == new_cell, model.states.loc.labels)

            if length(model.actions) == 1
                B_true[new_cell_id, loc_id, action[1]] = 1.0
            else
                # two actions
                B_true[new_cell_id, loc_id, action[1], action[2]] = 1.0
            end

        end
    end

    # for two actions, some sums in B matrix will be zero, as these are never used.
    @assert all(isapprox.(Statistics.sum(B_true, dims=1), 1.0) .|| isapprox.(Statistics.sum(B_true, dims=1), 0.0))
    @assert sum(B_true) == model.states.loc.B_dims[1] * 5  # 5 actual actions

    CONFIG = @set CONFIG.B_true = B_true
    
    #@infiltrate; @assert false
    return CONFIG
end