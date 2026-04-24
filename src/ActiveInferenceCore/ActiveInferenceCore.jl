"""
    ActiveInferenceCore

    
"""
module ActiveInferenceCore

#File with core types
include("core_types.jl")

#Export core functions and types
export AbstractGenerativeModel, 
    AbstractActionType, 
    AbstractObservationType,
    DiscreteObservations,
    ContinuousObservations,
    MixedObservations,
    NoObservations,
    ContinuousActions,
    MixedActions,
    NoActions
    AbstractInferenceEnvironment,
    AbstractInferenceActions,
    infer_environment,
    infer_actions,
    set_variables!
    active_inference!

end
