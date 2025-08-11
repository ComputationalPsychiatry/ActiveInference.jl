# @infiltrate; @assert false



####################################################################################################

function make_A(model, CONFIG) 
    
    A = model.obs.loc_obs.A
    
    # make the location observation only depend on the location state
    for ii in 1:model.obs.loc_obs.A_dims[1]
        A[ii, ii] = 1. 
    end

    #@infiltrate; @assert false
end