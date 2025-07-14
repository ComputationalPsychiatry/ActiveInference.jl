""" Constructor for the AIFAgent in the DiscretePOMDP module."""
function AIFAgent(;
    generative_model::GenerativeModel,
    perceptual_process::PerceptualProcess,
    perception::Function
)
    
    fill_missing_parameters(generative_model, perceptual_process)

    return AIFAgent(generative_model, perceptual_process, perception)
end



#### Below are functions for filling out missing parameters ####
function fill_missing_parameters(generative_model::GenerativeModel, perceptual_process::PerceptualProcess)

    # Provide the prior over states from the generative model to the perceptual_process
    perceptual_process.prior = generative_model.D

    # Create a default E parameter based on policy length from action_process
    # missing #

    # Create Prior over learned parameters if concentration parameter is given.
    create_learning_priors(
        generative_model,
        perceptual_process.A_learning,
        perceptual_process.B_learning,
        perceptual_process.D_learning
    )

    return
end 

function create_learning_priors(
    generative_model::GenerativeModel,
    A_learning::Union{Nothing, Learn_A},
    B_learning::Union{Nothing, Learn_B},
    D_learning::Union{Nothing, Learn_D}
)
    # Initialize priors for A, B, and D based on the learning settings
    if !isnothing(A_learning) && A_learning.prior == nothing
        A_learning.prior = deepcopy(generative_model.A) .* A_learning.concentration_parameter
    end

    if !isnothing(B_learning) && B_learning.prior == nothing
        B_learning.prior = deepcopy(generative_model.B) .* B_learning.concentration_parameter
    end

    if !isnothing(D_learning) && D_learning.prior == nothing
        D_learning.prior = deepcopy(generative_model.D) .* D_learning.concentration_parameter
    end

end