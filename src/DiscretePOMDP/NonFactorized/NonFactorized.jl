"""
NonFactorized submodule for DiscretePOMDP containing non-factorized implementations.
"""

module NonFactorized

# Import necessary packages
using ...ActiveInference: ReverseDiff, LogExpFunctions, LinearAlgebra
using ...ActiveInference.LogExpFunctions: softmax
using ...ActiveInference.LinearAlgebra: dot

# Import from parent modules
using ...ActiveInferenceCore: AbstractGenerativeModel, AbstractPerceptualProcess, AbstractActionProcess, AbstractOptimEngine, DiscreteActions, DiscreteObservations, DiscreteStates, AIFAgent


# Include generative model type and files
include("GenerativeModel/utils/GenerativeModelInfoStruct.jl")
include("GenerativeModel/utils/CheckGenerativeModel.jl")
include("GenerativeModel/GenerativeModel.jl")

# Include the perceptual process type and files
include("PerceptualProcess/learning/LearningStructs.jl")
include("PerceptualProcess/learning/learning_update_functions.jl")
include("PerceptualProcess/utils/PerceptualProcessInfoStruct.jl")
include("PerceptualProcess/optim_engines/FixedPointIteration.jl")
include("PerceptualProcess/utils/utils.jl")
include("PerceptualProcess/PerceptualProcess.jl")


# Include action process type and files
include("ActionProcess/ActionProcess.jl")
include("ActionProcess/utils/prediction_utils.jl")
include("ActionProcess/utils/posterior_policies_utils.jl")

# Include agent initialization
include("AgentInitialization/AgentInit.jl")

# Include utility functions
include("../../utils/maths.jl")
include("../../utils/utils.jl")

# Export main types and functions
export GenerativeModel, PerceptualProcess, ActionProcess
export Learn_A, Learn_B, Learn_D
export perception

end
