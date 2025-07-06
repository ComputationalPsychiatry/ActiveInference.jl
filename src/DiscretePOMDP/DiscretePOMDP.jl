module DiscretePOMDP

# Import the abstract type from the parent module
using ..ActiveInferenceCore: AbstractGenerativeModel

include("GenerativeModel.jl")
include("utils\\GenerativeModelUtils.jl")

# Including util functions from general utils folder
include("../utils/maths.jl")
include("../utils/utils.jl")

end