"""
    ActiveInferenceCore

    
"""
module ActiveInferenceCore

#File with core types
include("core_types.jl")
#File with core functions
include("core_functions.jl")

#Export core functions and types
export AbstractGenerativeModel,
    AIFModel,
    infer_environment,
    infer_actions,
    set_variables!,
    active_inference!
end
