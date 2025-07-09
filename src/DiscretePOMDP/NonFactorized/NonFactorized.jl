"""
NonFactorized submodule for DiscretePOMDP containing non-factorized implementations.
"""

module NonFactorized

# Import from parent modules
using ...ActiveInferenceCore: AbstractGenerativeModel, AbstractPerceptualProcess, DiscreteActions, DiscreteObservations, DiscreteStates

# Include generative model type and files
include("GenerativeModel/utils/GenerativeModelInfoStruct.jl")
include("GenerativeModel/utils/CheckGenerativeModel.jl")
include("GenerativeModel/GenerativeModel.jl")

# Include the perceptual process type and files
include("PerceptualProcess/utils/LearningStructs.jl")
include("PerceptualProcess/utils/PerceptualProcessInfoStruct.jl")
include("PerceptualProcess/utils/FixedPointIteration.jl")
include("PerceptualProcess/utils/utils.jl")
include("PerceptualProcess/PerceptualProcess.jl")


# Include action process type and files
include("ActionProcess.jl")

# Include utility functions
include("../../utils/maths.jl")
include("../../utils/utils.jl")

# Export main types and functions
export GenerativeModel, PerceptualProcess
export Learn_A, Learn_B, Learn_D

end
