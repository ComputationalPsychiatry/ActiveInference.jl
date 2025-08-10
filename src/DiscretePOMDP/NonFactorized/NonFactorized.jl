"""
NonFactorized submodule for DiscretePOMDP containing non-factorized implementations.
"""

module NonFactorized

# Import necessary packages
using ...ActiveInference: ReverseDiff
using ...ActiveInference.LogExpFunctions: softmax
using ...ActiveInference.LinearAlgebra: dot
using ...ActiveInference.Distributions: Multinomial

# Import from parent modules
using ...ActiveInferenceCore
import ...ActiveInferenceCore: AbstractGenerativeModel, AbstractPerceptualProcess, AbstractActionProcess, AbstractOptimEngine, DiscreteActions, DiscreteObservations, DiscreteStates, AIFModel


# Include generative model type and files
include("GenerativeModel/utils/GenerativeModelInfoStruct.jl")
include("GenerativeModel/utils/CheckGenerativeModel.jl")
include("GenerativeModel/utils/create_matrix_templates.jl")
include("GenerativeModel/GenerativeModel.jl")

# Include the perceptual process type and files
include("PerceptualProcess/learning/LearningStructs.jl")
include("PerceptualProcess/learning/learning_update_functions.jl")
include("PerceptualProcess/utils/OptimEngines.jl")
include("PerceptualProcess/utils/PerceptualProcessInfoStruct.jl")
include("PerceptualProcess/utils/utils.jl")
include("PerceptualProcess/PerceptualProcess.jl")

# Include action process type and files
include("ActionProcess/utils/ActionProcessInfoStruct.jl")
include("ActionProcess/ActionProcess.jl")
include("ActionProcess/utils/action_selection.jl")

# Include agent initialization
include("AgentInitialization/AgentInit.jl")

# Include perception function
include("PerceptionFunction/fixed_point_iteration.jl")

# Include prediction function
include("PredictionFunction/prediction_utils.jl")
include("PredictionFunction/prediction.jl")

# Include planning function
include("PlanningFunction/planning_function.jl")
include("PlanningFunction/planning_utils.jl")

# Include utility functions
include("../../utils/maths.jl")
include("../../utils/utils.jl")

# Export main types and functions
export GenerativeModel, PerceptualProcess, ActionProcess
export Learn_A, Learn_B, Learn_D
# export perception

end
