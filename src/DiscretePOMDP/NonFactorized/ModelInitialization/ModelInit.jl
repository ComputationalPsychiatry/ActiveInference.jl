""" Constructor for the AIFModel in the DiscretePOMDP module."""
function ActiveInferenceCore.AIFModel(;
    generative_model::GenerativeModel,
    perceptual_process::CAVI,
    action_process::ActionProcess
)
    
    fill_missing_parameters(generative_model, perceptual_process, action_process);

    return AIFModel(generative_model, perceptual_process, action_process)
end

# Simple Constructor
function ActiveInferenceCore.AIFModel(
    A::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N},
    B::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N};
    C::Union{Vector{Vector{T}}, Nothing} where {T <: Real} = nothing,
    D::Union{Vector{Vector{T}}, Nothing} where {T <: Real} = nothing,
    A_learning::Union{Nothing, Learn_A} = nothing,
    B_learning::Union{Nothing, Learn_B} = nothing,
    D_learning::Union{Nothing, Learn_D} = nothing,
    num_iter::Int = 10,
    dF_tol::Float64 = 1e-3,
    use_utility::Bool = true,
    use_states_info_gain::Bool = true,
    use_param_info_gain::Bool = false,
    gamma::Real = 16.0,
    E::Union{Vector{T}, Nothing} where {T <: Real} = nothing,
    policy_length::Int = 2,
    policies::Union{Vector{Matrix{Int64}}, Nothing} = nothing,
    previous_action::Union{Vector{Int}, Nothing} = nothing,
    posterior_policies::Union{Vector{Float64}, Nothing} = nothing,
    expected_free_energy::Union{Vector{Float64}, Nothing} = nothing,
    action_selection::Symbol = :stochastic,
    alpha::Real = 16.0
)

    # Create the generative model
    generative_model = GenerativeModel(A=A, B=B, C=C, D=D, verbose=true);

    # Create the perceptual process
    perceptual_process = CAVI(
        A_learning = A_learning,
        B_learning = B_learning,
        D_learning = D_learning,
        num_iter = num_iter,
        dF_tol = dF_tol,
        verbose = true
    );

    # Create the action process
    action_process = ActionProcess(
        use_utility = use_utility,
        use_states_info_gain = use_states_info_gain,
        use_param_info_gain = use_param_info_gain,
        gamma = gamma,
        E = E,
        policy_length = policy_length,
        policies = policies,
        previous_action = previous_action,
        posterior_policies = posterior_policies,
        expected_free_energy = expected_free_energy,
        action_selection = action_selection,
        alpha = alpha,
        verbose = true
    );

    return AIFModel(
        generative_model = generative_model, 
        perceptual_process = perceptual_process,
        action_process = action_process
    )
end
