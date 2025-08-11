# @infiltrate; @assert false



####################################################################################################

function make_A(model, CONFIG) 
    
    A1 = model.obs.loc_obs.A
    
    # make the location observation only depend on the location state
    for ii in 1:model.obs.loc_obs.A_dims[1]
        A1[ii, ii] = 1. 
    end

    # "Wall" Modality
    A2 = model.obs.wall_obs.A
    MAZE_vec = vec(model.states.loc.extra.MAZE)
    A2[:,:] = vcat((1 .- MAZE_vec'), MAZE_vec')
    #heatmap(A[2], color=cgrad(:greys), yflip=true, yticks=1:2,xticks=1:8:81)

    #@infiltrate; @assert false
end