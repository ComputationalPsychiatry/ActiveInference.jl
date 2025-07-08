module DiscretePOMDP

# Import the abstract type from the parent module
using ..ActiveInferenceCore: AbstractGenerativeModel, AbstractPerceptualProcess

# Generative Model
include("GenerativeModel\\utils\\GenerativeModelInfoStruct.jl")
include("GenerativeModel\\utils\\CheckGenerativeModel.jl")
include("GenerativeModel\\GenerativeModel.jl")

# Perceptual Process
include("PerceptualProcess\\utils\\LearningStructs.jl")
include("PerceptualProcess\\utils\\LearningUtils.jl")
include("PerceptualProcess\\PerceptualProcess.jl")

# Including util functions from general utils folder
include("../utils/maths.jl")
include("../utils/utils.jl")

end