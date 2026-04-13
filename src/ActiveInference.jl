module ActiveInference

#Read core module
include("ActiveInferenceCore/ActiveInferenceCore.jl")

#Read module with utilities etc
include("ActiveInferenceBase/ActiveInferenceBase.jl")

#Read module with action planning functions
include("ActiveInferenceActionPlanning/ActiveInferenceActionPlanning.jl")

end
