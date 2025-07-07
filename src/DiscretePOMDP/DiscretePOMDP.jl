module DiscretePOMDP

# Import the abstract type from the parent module
using ..ActiveInferenceCore: AbstractGenerativeModel, AbstractPerceptualProcess

# Generative Model
include("GenerativeModel\\utils\\GenerativeModelInfoStruct.jl")
include("GenerativeModel\\utils\\GenerativeModelUtils.jl")
include("GenerativeModel\\GenerativeModel.jl")

# Perceptual Process
include("PerceptualProcess\\utils\\LearningStructs.jl")
include("PerceptualProcess\\utils\\learning_utils.jl")
include("PerceptualProcess\\PerceptualProcess.jl")

# Including util functions from general utils folder
include("../utils/maths.jl")
include("../utils/utils.jl")

end