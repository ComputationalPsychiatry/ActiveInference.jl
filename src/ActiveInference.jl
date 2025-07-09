module ActiveInference

using ActionModels
using LinearAlgebra
using IterTools
using Random
using Distributions
using LogExpFunctions
using ReverseDiff
using Parameters

include("utils/maths.jl")
include("pomdp/struct.jl")
include("pomdp/struct_utils.jl")
include("pomdp/learning.jl")
include("utils/utils.jl")
include("pomdp/inference.jl")
include("ActionModelsExtensions/get_states.jl")
include("ActionModelsExtensions/get_parameters.jl")
include("ActionModelsExtensions/get_settings.jl")
include("ActionModelsExtensions/get_history.jl")
include("ActionModelsExtensions/set_parameters.jl")
include("ActionModelsExtensions/reset.jl")
include("ActionModelsExtensions/give_inputs.jl")
include("ActionModelsExtensions/set_save_history.jl")
include("pomdp/POMDP.jl")
include("utils/helper_functions.jl")
include("utils/create_matrix_templates.jl")

# Include the AIFCore module first
include("AIFCore/AIFCore.jl")
using .ActiveInferenceCore

# Include the DiscretePOMDP module
include("DiscretePOMDP/DiscretePOMDP.jl")
using .DiscretePOMDP

export # utils/create_matrix_templates.jl
        create_matrix_templates,

       # AIFCore module
       AbstractGenerativeModel,
       DiscreteActions,
       DiscreteObservations, 
       DiscreteStates,
       ContinuousActions,
       ContinuousObservations,
       ContinuousStates,
       MixedActions,
       MixedObservations,
       MixedStates,
       AIFAgent,
       active_inference,

       # DiscretePOMDP module
       DiscretePOMDP,
       init_generative_model,

       # struct.jl
       init_pomdp_aif_settings,
       init_pomdp_aif_parameters,
       init_pomdp_aif,
       infer_states!,
       infer_policies!,
       sample_action!,
       update_parameters!,

       # POMDP.jl
       action_pomdp!,

       # ActionModelsExtensions
       get_states,
       get_parameters,
       get_settings,
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






