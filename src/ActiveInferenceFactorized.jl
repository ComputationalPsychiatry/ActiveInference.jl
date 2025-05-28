module ActiveInferenceFactorized

using ActionModels
using LinearAlgebra
using IterTools
using Random
using Distributions
using LogExpFunctions
using ReverseDiff

include("factorized/utils/maths.jl")
include("factorized/struct.jl")
include("factorized/sophisticated.jl")
include("factorized/learning.jl")
include("factorized/utils/utils.jl")
include("factorized/inference.jl")
include("ActionModelsExtensions/get_states.jl")
include("ActionModelsExtensions/get_parameters.jl")
include("ActionModelsExtensions/get_history.jl")
include("ActionModelsExtensions/set_parameters.jl")
include("ActionModelsExtensions/reset.jl")
include("ActionModelsExtensions/give_inputs.jl")
include("ActionModelsExtensions/set_save_history.jl")
include("factorized/POMDP.jl")
include("factorized/utils/helper_functions.jl")
include("factorized/utils/create_matrix_templates.jl")

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
       action_factorized!,

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
    include("factorized/utils/maths.jl")

    export TMazeEnv, step_TMaze!, reset_TMaze!, initialize_gp
       
    end
end






