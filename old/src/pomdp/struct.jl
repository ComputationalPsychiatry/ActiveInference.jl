""" Agent Structs

This module contains the structs, initialization functions, and API functions for the POMDPActiveInference struct.

It is structured as follows:
- Structs
    - Settings Struct
    - Parameters Struct
    - States Struct
    - History Struct
    - POMDP Active Inference Struct

- Initialization Functions
    - init_pomdp_aif_settings
    - init_pomdp_aif_parameters
    - init_pomdp_aif

- API Functions
    - infer_states!
    - infer_policies!
    - sample_action!
    - update_parameters!
"""

################## Structs ##################

# Abstract top level struct
abstract type ActionModel end

""" ---- Settings Struct ---- """
@with_kw mutable struct POMDPActiveInferenceSettings

    policy_length::Int64 = 1
    use_utility::Bool = true
    use_states_info_gain::Bool = true
    use_param_info_gain::Bool = false
    action_selection::String = "stochastic"
    modalities_to_learn::Union{Vector{Int64}, String} = "all"
    factors_to_learn::Union{Vector{Int64}, String} = "all"
    FPI_n_iter::Int64 = 10
    FPI_tol::Float64 = 0.001
    save_history::Bool = true
    policies::Union{Vector{Matrix{Int64}}, Nothing} = nothing

    # Internal fields
    _n_states::Union{Vector{Int64}, Nothing} = nothing
    _n_observations::Union{Vector{Int64}, Nothing} = nothing
    _n_controls::Union{Vector{Int64}, Nothing} = nothing
    _control_fac_idx::Union{Vector{Int64}, Nothing}  = nothing

end

""" ---- Parameters Struct ---- """
@with_kw mutable struct POMDPActiveInferenceParameters

    # Generative model parameters
    A::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} = nothing # A-matrix
    B::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} = nothing # B-matrix
    C::Union{Vector{Vector{T}}, Nothing} where {T <: Real} = nothing # C-vectors
    D::Union{Vector{Vector{T}}, Nothing} where {T <: Real} = nothing # D-vectors
    E::Union{Vector{T}, Nothing} where {T <: Real} = nothing # E-vector (Habits)
    pA::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} = nothing# Dirichlet priors for A-matrix
    pB::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} = nothing # Dirichlet priors for B-matrix
    pD::Union{Vector{Vector{T}}, Nothing} where {T <: Real} = nothing # Dirichlet priors for D-vector
    gamma::Real = 16.0
    alpha::Real = 16.0
    lr_pA::Real = 1.0
    fr_pA::Real = 1.0
    lr_pB::Real = 1.0
    fr_pB::Real = 1.0
    lr_pD::Real = 1.0
    fr_pD::Real = 1.0

end

""" ---- States Struct ---- """
mutable struct POMDPActiveInferenceStates

    qs_current::Vector{Vector{T}} where {T <: Real}
    obs_current::Vector{T} where {T <: Real}
    prior::Vector{Vector{T}} where {T <: Real}
    q_pi::Vector{T} where {T <: Real}
    G::Vector{T} where {T <: Real}
    action::Vector{Int}
    SAPE::Union{Vector{T}, Missing} where T <:Real
    bayesian_model_averages::Union{Vector{Vector{T}}, Missing} where T <: Real

end

""" ---- History Struct ---- """
mutable struct POMDPActiveInferenceHistory

    qs_current::Union{Vector{Vector{Vector{T}}}, Missing} where {T <: Real}
    obs_current::Union{Vector{Vector{T}}, Missing} where {T <: Real}
    prior::Union{Vector{Vector{Vector{T}}}, Missing} where {T <: Real}
    q_pi::Union{Vector{Vector{T}}, Missing} where {T <: Real}
    G::Union{Vector{Vector{T}}, Missing} where {T <: Real}
    action::Union{Vector{Vector{T}}, Missing} where {T <: Real}
    SAPE::Union{Vector{Vector{T}}, Missing} where {T <:Real}
    bayesian_model_averages::Union{Vector{Vector{Vector{T}}}, Missing} where T <: Real

end

""" ---- POMDP Active Inference Struct ---- """
mutable struct POMDPActiveInference <: ActionModel

    parameters::POMDPActiveInferenceParameters
    settings::POMDPActiveInferenceSettings
    states::POMDPActiveInferenceStates
    history::POMDPActiveInferenceHistory

end

################## Initialization functions ##################
"""
    init_pomdp_aif_settings(; kwargs...)

Create an instance of the POMDPActiveInferenceSettings struct.

# Arguments
- `kwargs...`: Keyword arguments to set the fields of the POMDPActiveInferenceSettings struct. See POMDPActiveInferenceSettings struct.

"""
function init_pomdp_aif_settings(; kwargs...)

    # Create an instance of the settings struct
    settings = POMDPActiveInferenceSettings(; kwargs...)

    # Checking settings and making sure they are formatted correctly
    check_settings(settings)
    
    # Return an instance of the pomdp aif settings struct
    return settings
end

"""
    init_pomdp_aif_parameters(; kwargs...)

Create an instance of the POMDPActiveInferenceParameters struct.

# Arguments
- `kwargs...`: Keyword arguments to set the fields of the POMDPActiveInferenceParameters struct. See POMDPActiveInferenceParameters struct.

"""
function init_pomdp_aif_parameters(; kwargs...)
    
    # Create an instance of the parameter struct
    parameters = POMDPActiveInferenceParameters(; kwargs...)

    # Perform checks using the fields from parameter struct
    check_parameters(parameters)

    # Return an instance of the pomdp aif parameters struct
    return parameters
end

"""
    init_pomdp_aif(; parameters::POMDPActiveInferenceParameters, settings::POMDPActiveInferenceSettings, verbose::Bool = true)

Create an instance of the POMDPActiveInference struct.

# Arguments
- `parameters::POMDPActiveInferenceParameters`: An instance of the POMDPActiveInferenceParameters struct.
- `settings::POMDPActiveInferenceSettings`: An instance of the POMDPActiveInferenceSettings struct.
- `verbose::Bool`: A boolean flag to print warnings if parameters are not provided and has been inferred.

"""
function init_pomdp_aif(;
    parameters::POMDPActiveInferenceParameters = POMDPActiveInferenceParameters(),
    settings::POMDPActiveInferenceSettings = POMDPActiveInferenceSettings(), verbose::Bool = true)

    # Check if the parameters and settings are compatable
    check_settings_and_parameters(settings, parameters)

    # Inferring undefined parameters and printing warnings if verbose is true
    infer_missing_parameters(parameters, settings, verbose)

    # Inferring undefined settings
    infer_missing_settings(settings, parameters)

    # Construct a states struct
    states_struct = construct_states_struct(parameters, settings)

    # Construct a history struct
    history_struct = construct_history_struct(states_struct)
    
    return POMDPActiveInference(parameters, settings, states_struct, history_struct)
end

################## API functions ##################

""" Update the agents's beliefs over states """
function infer_states!(aif::POMDPActiveInference, obs::Vector{Int64})
    if !isempty(aif.states.action)
        int_action = round.(Int, aif.states.action)
        aif.states.prior = get_states_prediction(aif.states.qs_current, aif.parameters.B, reshape(int_action, 1, length(int_action)))[1]
    else
        aif.states.prior = aif.parameters.D
    end

    # Update posterior over states
    aif.states.qs_current = update_posterior_states(aif.parameters.A, obs, prior=aif.states.prior, num_iter=aif.settings.FPI_n_iter, dF_tol=aif.settings.FPI_tol)

    # Adding the obs to the agent struct
    aif.states.obs_current = obs

    # Push changes to agent's history
    push!(aif.history.prior, aif.states.prior)
    push!(aif.history.qs_current, aif.states.qs_current)
    push!(aif.history.obs_current, aif.states.obs_current)

    return aif.states.qs_current
end

""" Update the agents's beliefs over policies """
function infer_policies!(aif::POMDPActiveInference)
    # Update posterior over policies and expected free energies of policies
    q_pi, G = update_posterior_policies(aif.states.qs_current, aif.parameters.A, aif.parameters.B, aif.parameters.C, aif.settings.policies, aif.settings.use_utility, aif.settings.use_states_info_gain, aif.settings.use_param_info_gain, aif.parameters.pA, aif.parameters.pB, aif.parameters.E, aif.parameters.gamma)

    aif.states.q_pi = q_pi
    aif.states.G = G  

    # Push changes to agent's history
    push!(aif.history.q_pi, copy(aif.states.q_pi))
    push!(aif.history.G, copy(aif.states.G))

    return q_pi
end

""" Sample action from the beliefs over policies """
function sample_action!(aif::POMDPActiveInference)
    action = sample_action(aif.states.q_pi, aif.settings.policies, aif.settings._n_controls; action_selection=aif.settings.action_selection, alpha=aif.parameters.alpha)

    aif.states.action = action 

    # Push action to agent's history
    push!(aif.history.action, copy(action))

    return action
end

""" Update the agent's beliefs over states and observations """
function update_parameters!(aif::POMDPActiveInference)

    if aif.parameters.pA != nothing
        update_A(aif)
    end

    if aif.parameters.pB != nothing
        update_B(aif)
    end

    if aif.parameters.pD != nothing
        update_D(aif)
    end
    
end
