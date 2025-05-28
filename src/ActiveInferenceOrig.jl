module ActiveInferenceOrig

using ActionModels
using LinearAlgebra
using IterTools
using Random
using Distributions
using LogExpFunctions
using ReverseDiff

include("utils/maths.jl")
include("pomdp/struct.jl")
include("pomdp/learning.jl")
include("utils/utils.jl")
include("pomdp/inference.jl")
include("ActionModelsExtensions/get_states.jl")
include("ActionModelsExtensions/get_parameters.jl")
include("ActionModelsExtensions/get_history.jl")
include("ActionModelsExtensions/set_parameters.jl")
include("ActionModelsExtensions/reset.jl")
include("ActionModelsExtensions/give_inputs.jl")
include("ActionModelsExtensions/set_save_history.jl")
include("pomdp/POMDP.jl")
include("utils/helper_functions.jl")
include("utils/create_matrix_templates.jl")

export # utils/create_matrix_templates.jl
        create_matrix_templates,
       
       # utils/maths.jl
       normalize_distribution,
       softmax_array,
       normalize_arrays,

       # utils/utils.jl
       array_of_any_zeros, 
       onehot,
       get_model_dimensions,

       # struct.jl
       init_aif,
       infer_states!,
       infer_policies!,
       sample_action!,
       update_A!,
       update_B!,
       update_D!,
       update_parameters!,
       MetaModel,

       # POMDP.jl
       action_pomdp!,

       # ActionModelsExtensions
       get_states,
       get_parameters,
       get_history,
       set_parameters!,
       reset!,
       single_input!,
       give_inputs!,
       set_save_history!

    module Environments

    using LinearAlgebra
    using ActiveInference
    using Distributions
    
    include("Environments/EpistChainEnv.jl")
    
    export EpistChainEnv, step!, reset_env!

    include("Environments/TMazeEnv.jl")
    include("utils/maths.jl")

    export TMazeEnv, step_TMaze!, reset_TMaze!, initialize_gp
       
    end
end






