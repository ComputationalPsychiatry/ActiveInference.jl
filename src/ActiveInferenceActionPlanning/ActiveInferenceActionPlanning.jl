#TODO: 
#- Add probability weighting on costs (for observation nodes, get probabilities from the predictive prosterior)
#- Figure out whether to store results for the branches separately



module ActiveInferenceActionPlanning
    
    #Functionality general across tree search types
    include("base_functionality.jl")
    
    #The full recursive tree search method
    include("full_recursive_tree_search.jl")

end
