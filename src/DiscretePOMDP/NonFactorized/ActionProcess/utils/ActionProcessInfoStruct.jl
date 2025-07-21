"""
InfoStruct for tracking action process configuration and policy settings.
"""

mutable struct ActionProcessInfo

    # Utility and information gain flags
    use_utility::Bool
    use_states_info_gain::Bool
    use_param_info_gain::Bool

    # Policy configuration
    policy_length::Int
    n_policies::Int
    policies_provided::Bool

    # Prior preferences configuration
    E_provided::Bool
    gamma::Real

    # Only if action is enabled
    action_selection::String
    alpha::Real

    function ActionProcessInfo(
        use_utility::Bool,
        use_states_info_gain::Bool,
        use_param_info_gain::Bool,
        policy_length::Int,
        policies::Union{Vector{Matrix{Int64}}, Nothing},
        E::Union{Vector{T}, Nothing} where {T <: Real},
        gamma::Real,
        action_selection::String,
        alpha::Real
    )
        
        # Policy information
        policies_provided = !isnothing(policies)
        n_policies = policies_provided ? length(policies) : 0

        # Prior preferences information
        E_provided = !isnothing(E)

        new(
            use_utility,
            use_states_info_gain,
            use_param_info_gain,
            policy_length,
            n_policies,
            policies_provided,
            E_provided,
            gamma,
            action_selection,
            alpha
        )
    end
end

"""
Pretty print function for ActionProcessInfo.
"""
function show_info(info::ActionProcessInfo; verbose::Bool = true)
    if !verbose
        return
    end
    
    println("\n" * "="^100)
    println("🕹️  Action Process Information")
    println("="^100)

    println("\n🔧  Policy Configuration:")
    println("   • Policy length: $(info.policy_length)")
    println("   • Number of policies: $(info.n_policies)")
    println("   • Policies provided: $(info.policies_provided)")
    
    println("\n📊 EFE Calculation Configuration:")
    println("   • Use utility: $(info.use_utility)")
    println("   • Use states info gain: $(info.use_states_info_gain)")
    println("   • Use parameter info gain: $(info.use_param_info_gain)")

    println("\n🧮 Prior Configuration:")
    println("   • Prior over policies (E) provided: $(info.E_provided)")
    println("   • Gamma (precision): $(info.gamma)")
    
    println("="^100)
end