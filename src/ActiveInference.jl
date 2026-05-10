module ActiveInference
using Reexport

#Read core module with types and core functions
include("ActiveInferenceCore/ActiveInferenceCore.jl")
@reexport using .ActiveInferenceCore

#Read module with utilities etc
include("ActiveInferenceBase/ActiveInferenceBase.jl")
@reexport using .ActiveInferenceBase

#Read module with action planning functions
include("ActiveInferenceActionPlanning/ActiveInferenceActionPlanning.jl")
@reexport using .ActiveInferenceActionPlanning

export active_inference!

end
