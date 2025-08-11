"""
In this script, we define a the action_process process for the DiscretePOMDP.
"""

using ..ActiveInferenceCore: AbstractActionProcess, AIFModel

# Struct for containing the action process
mutable struct ActionProcess <: AbstractActionProcess

    # Struct for containing the "meta" information, such as whether to update parameters etc
    info::ActionProcessInfo

    # Settings for the actions process
    use_utility::Bool
    use_states_info_gain::Bool
    use_param_info_gain::Bool
    gamma::Real #? should not be here?

    # Field containing prior over policies 'E'. Also called 'habits'.
    E::Union{Vector{T}, Nothing} where {T <: Real}

    # Fields for containing information about the action process
    policy_length::Int
    policies::Union{Vector{Matrix{Int64}}, Nothing}

    # Fields containing predictions, actions, and posterior policies
    action::Union{Vector{Int}, Nothing}
    posterior_policies::Union{Vector{Float64}, Nothing}
    expected_free_energy::Union{Vector{Float64}, Nothing}

    # Field for action selection
    action_selection::String
    alpha::Real

    function ActionProcess(;
        use_utility::Bool = true,
        use_states_info_gain::Bool = true,
        use_param_info_gain::Bool = false,
        gamma::Real = 16.0,
        E::Union{Vector{T}, Nothing} where {T <: Real} = nothing,
        policy_length::Int = 2,
        policies::Union{Vector{Matrix{Int64}}, Nothing} = nothing,
        action::Union{Vector{Int}, Nothing} = nothing,
        posterior_policies::Union{Vector{Float64}, Nothing} = nothing,
        expected_free_energy::Union{Vector{Float64}, Nothing} = nothing,
        action_selection::String = "stochastic",
        alpha::Real = 16.0,
        verbose::Bool = true
    )

        # Create the ActionProcessInfo struct
        info = ActionProcessInfo(
            use_utility,
            use_states_info_gain,
            use_param_info_gain,
            policy_length,
            policies,
            E,
            gamma,
            action_selection,
            alpha
        )

        show_info(info, verbose=verbose)

        if action_selection != "stochastic" && action_selection != "deterministic"
            error("action_selection must be either 'stochastic' or 'deterministic'")
        end

        new(
            info, 
            use_utility, 
            use_states_info_gain, 
            use_param_info_gain, 
            gamma, 
            E, 
            policy_length, 
            policies, 
            action, 
            posterior_policies, 
            expected_free_energy, 
            action_selection,
            alpha
        )
    end
end
