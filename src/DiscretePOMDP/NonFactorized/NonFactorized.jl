"""
NonFactorized submodule for DiscretePOMDP containing non-factorized implementations.
"""

module NonFactorized

# Import from parent modules
using ...ActiveInferenceCore: AbstractGenerativeModel, AbstractPerceptualProcess, AbstractOptimEngine, DiscreteActions, DiscreteObservations, DiscreteStates, AIFAgent

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

# Include agent initialization
include("AgentInitialization/AgentInit.jl")

# Include utility functions
include("../../utils/maths.jl")
include("../../utils/utils.jl")

# Export main types and functions
export GenerativeModel, PerceptualProcess
export Learn_A, Learn_B, Learn_D
export perception

end
