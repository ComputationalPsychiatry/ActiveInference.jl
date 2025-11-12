module DiscretePOMDP

# Import the abstract types from the parent module
using ..ActiveInferenceCore
using ..ActiveInferenceCore: AbstractGenerativeModel, AbstractPerceptualProcess, DiscreteActions, DiscreteObservations, DiscreteStates


# Include the Factorized submodule
# include("Factorized/Factorized.jl")
# using .Factorized

# (NOTE SAM: Revisit the NonFactroized version later)
#Include the NonFactorized submodule
include("NonFactorized/NonFactorized.jl")
using .NonFactorized

# Re-export the NonFactorized module's exports so they can be accessed as DiscretePOMDP.GenerativeModel
export GenerativeModel, PerceptualProcess, Learn_A, Learn_B, Learn_D

# Including general util functions that might be shared across different factorization types
include("../utils/maths.jl")
include("../utils/utils.jl")

end