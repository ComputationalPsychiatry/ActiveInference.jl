# @infiltrate; @assert false



####################################################################################################

function make_A(model, CONFIG) 
    
    A1 = model.obs.loc_obs.A
    
    # make the location observation only depend on the location state
    for ii in 1:model.obs.loc_obs.A_dims[1]
        A1[ii, ii] = 1. 
    end
    #heatmap(A[1], color=cgrad(:greys), yflip=true, aspect_ratio=:equal)

    #=
    # Stochastic A-matrix
    A[1] += 0.8 * I(81)
    A[1] .+= (1.0 - 0.8) / 81
    heatmap(A[1], color=cgrad(:greys), yflip=true, aspect_ratio=:equal)
    =#

    # "Wall" Modality
    A2 = model.obs.wall_obs.A
    MAZE_vec = vec(model.states.loc.extra.MAZE)
    A2[:,:] = vcat((1 .- MAZE_vec'), MAZE_vec')
    #heatmap(A[2], color=cgrad(:greys), yflip=true, yticks=1:2,xticks=1:8:81)

    # Preference Modality
    # Probabilities of observing Safe Location
    A3 = model.obs.safe_obs.A
    A3[1, :] = vec(model.obs.safe_obs.extra.PREFERENCES)

    # Probabilities of observing Aversive Location
    A3[2, :] = 1 .- vec(model.obs.safe_obs.extra.PREFERENCES)
    #heatmap(A[3], color=cgrad(:greys), yflip=true)

    #@infiltrate; @assert false
end