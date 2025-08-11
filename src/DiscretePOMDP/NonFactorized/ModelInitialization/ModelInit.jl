""" Constructor for the AIFModel in the DiscretePOMDP module."""
function ActiveInferenceCore.AIFModel(;
    generative_model::GenerativeModel,
    perceptual_process::PerceptualProcess,
    action_process::ActionProcess
)
    
    fill_missing_parameters(generative_model, perceptual_process, action_process)

    return AIFModel(generative_model, perceptual_process, action_process)
end


#### Below are functions for filling out missing parameters ####
function fill_missing_parameters(generative_model::GenerativeModel, perceptual_process::PerceptualProcess, action_process::ActionProcess)

    # Provide the prior over states from the generative model to the perceptual_process
    perceptual_process.prior_qs_prediction = generative_model.D

    # Create Prior over learned parameters if concentration parameter is given.
    create_learning_priors(
        generative_model,
        perceptual_process.A_learning,
        perceptual_process.B_learning,
        perceptual_process.D_learning
    )

    # Create policies if not provided
    if isnothing(action_process.policies)
        
        action_process.policies = construct_policies(
            generative_model.info.controls_per_factor, 
            action_process.policy_length
        )
        
    end

    # Create a default E parameter based on policy length from action_process
    n_policies = length(action_process.policies)
    action_process.E = fill(1.0 / n_policies, n_policies)

    # set default optim engine if not provided
    if perceptual_process.optim_engine === missing
        perceptual_process.optim_engine = FixedPointIteration()
    end

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

""" Function to create the policies of the model based on the generative model."""
function construct_policies(n_controls::Vector{Int}, policy_length::Int)

    # Create a vector of possible actions for each time step
    x = repeat(n_controls, policy_length)

    # Generate all combinations of actions across all time steps
    policies = collect(Iterators.product([1:i for i in x]...))

    # Initialize an empty vector to store transformed policies
    transformed_policies = Vector{Matrix{Int64}}()

    for policy_tuple in policies
        # Convert tuple into a vector
        policy_vector = collect(policy_tuple)
        
        # Reshape the policy vector into a matrix and transpose it
        policy_matrix = reshape(policy_vector, (length(policy_vector) ÷ policy_length, policy_length))'
        
        # Push the reshaped matrix to the vector of transformed policies
        push!(transformed_policies, policy_matrix)
    end

    return transformed_policies
end