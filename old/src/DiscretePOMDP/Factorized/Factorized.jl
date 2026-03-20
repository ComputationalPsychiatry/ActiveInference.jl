"""
Factorized submodule for DiscretePOMDP containing factorized implementation.
"""

module Factorized

# Import necessary packages
using ...ActiveInference: ReverseDiff
using ...ActiveInference.LogExpFunctions: softmax
using ...ActiveInference.LinearAlgebra: dot
using ...ActiveInference.Distributions: Multinomial

# Import from parent modules
using ...ActiveInferenceCore
import ...ActiveInferenceCore: AbstractGenerativeModel, AbstractPerceptualProcess, AbstractActionProcess, DiscreteActions, DiscreteObservations, DiscreteStates, AIFModel


# Include generative model type and files
include("GenerativeModel/utils/GenerativeModelInfoStruct.jl")
include("GenerativeModel/utils/CheckGenerativeModel.jl")
include("GenerativeModel/utils/create_matrix_templates.jl")
include("GenerativeModel/GenerativeModel.jl")

# Include the perceptual process type and files
include("PerceptualProcess/CAVI/learning/LearningStructs.jl")
include("PerceptualProcess/CAVI/CAVIInfoStruct.jl")
include("PerceptualProcess/CAVI/CAVI.jl")

# Include action process type and files
include("ActionProcess/utils/ActionProcessInfoStruct.jl")
include("ActionProcess/ActionProcess.jl")

# Include model initialization
include("ModelInitialization/ModelInit.jl")
include("ModelInitialization/fill_missing_parameters/CAVI_init.jl")

# Include perception function
include("PerceptionFunction/CAVI_perception/CAVI_function.jl")
include("PerceptionFunction/CAVI_perception/Learning/CAVI_learning.jl")
include("PerceptionFunction/CAVI_perception/Learning/learning_update_functions.jl")

# Include prediction function
include("PredictionFunction/prediction_utils.jl")
include("PredictionFunction/prediction_function.jl")

# Include planning function
include("PlanningFunction/planning_function.jl")
include("PlanningFunction/planning_utils.jl")
include("PlanningFunction/prediction_utils.jl")

# Include selection function
include("SelectionFunction/selection_function.jl")

# Include storage function
include("StoreBeliefsFunction/store_beliefs_function.jl")

# Include utility functions
include("../../utils/maths.jl")
include("../../utils/utils.jl")

# Export main types and functions
export GenerativeModel, CAVI, ActionProcess
export Learn_A, Learn_B, Learn_D
# export perception

end
