module ActiveInference

#Read core module with types and core functions
include("ActiveInferenceCore/ActiveInferenceCore.jl")
using .ActiveInferenceCore

#Read module with utilities etc
include("ActiveInferenceBase/ActiveInferenceBase.jl")
using .ActiveInferenceBase

#Read module with action planning functions
include("ActiveInferenceActionPlanning/ActiveInferenceActionPlanning.jl")
using .ActiveInferenceActionPlanning

export active_inference!

end
