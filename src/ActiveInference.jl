module ActiveInference

using ActionModels
using LinearAlgebra
using IterTools
using Random
using Distributions
using LogExpFunctions
using ReverseDiff
using Parameters
using Setfield: @lens, set, get

# Include the AIFCore module first
include("ActiveInferenceCore/ActiveInferenceCore.jl")
using .ActiveInferenceCore
export AIFModel, active_inference, active_inference_action, perception, policy_predictions, planning, selection, store_beliefs!

# Include the DiscretePOMDP module
include("DiscretePOMDP/DiscretePOMDP.jl")
using .DiscretePOMDP

import ActionModels: initialize_attributes
import ActionModels: store_action!

include("utils/maths.jl")
include("pomdp/struct.jl")
include("pomdp/struct_utils.jl")
include("pomdp/learning.jl")
include("utils/utils.jl")
include("pomdp/inference.jl")
include("ActionModelsExtensions/get_states.jl")
include("ActionModelsExtensions/model_attributes.jl")
include("ActionModelsExtensions/reset.jl")
include("ActionModelsExtensions/set_parameters.jl")
include("ActionModelsExtensions/store_action.jl")
include("pomdp/POMDP.jl")
include("utils/helper_functions.jl")
include("utils/create_matrix_templates.jl")



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
       AIFModel,
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

       get_state_types,
       get_statesm
       fieldname_in_type,
       initialize_attributes,
       reset!,
       reset_state!,
       set_parameters!,
       store_action!



       # ActionModelsExtensions
       

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






