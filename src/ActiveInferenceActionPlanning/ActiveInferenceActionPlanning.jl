#TODO: 
#- Add probability weighting on costs (for observation nodes, get probabilities from the predictive prosterior)
#- Figure out whether to store results for the branches separately, or how to do the backwards pass to get the action posterior
#_ make sure everything is DualNumber compatible


module ActiveInferenceActionPlanning
    
    #Functionality general across tree search types
    include("base_functionality.jl")
    
    #The full recursive tree search method
    include("full_recursive_tree_search.jl")

end
