""" -------- Struct Utility Functions -------- 

This file contains utility functions that are called directly in the structs.jl file for the POMDP Active Inference struct.

"""

"""
check generative model parameters

# Arguments
- `parameters::POMDPActiveInferenceParameters`

Throws an error if the generative model parameters are not valid:
- Both A and B must be provided.
- The dimensions of the matrices must be consistent.
- The values must be non-negative (except for C).
- The sum of each column or vector must be approximately 1.
- Not both the parameter and their prior can be provided.
"""
function check_parameters(parameters::POMDPActiveInferenceParameters)

    # Destructures the parameters of the parameter struct
    (; A, B, C, D, E, pA, pB, pD, gamma, alpha, lr_pA, fr_pA, lr_pB, fr_pB, lr_pD, fr_pD) = parameters

    # Check if A and/or pA has been provided.
    if isnothing(A) && isnothing(pA)
        throw(ArgumentError("A or pA must be provided in order to infer structure of the generative model."))

    elseif !isnothing(A) && !isnothing(pA)
        throw(ArgumentError("Both A and pA cannot be provided. Provide either A or pA."))
    end

    # Check if B and/or pB has been provided.
    if isnothing(B) && isnothing(pB)
        throw(ArgumentError("B or pB must be provided in order to infer structure of the generative model."))

    elseif !isnothing(B) && !isnothing(pB)
        throw(ArgumentError("Both B and pB cannot be provided. Provide either B or pB."))
    end

    # Check if both D and pD has been provided.
    if !isnothing(D) && !isnothing(pD)
        throw(ArgumentError("Both D and pD cannot be provided. Provide either D or pD."))
    end

    # Check if the number of states in A, B, and D are consistent.
    # We let this check be done on either the prior or the parameter, depending on which is provided.
    check_parameter_states(parameters)

    # Check if the number of observation modalities in A and C are consistent.
    # We let this check be done on either the prior or the parameter, depending on which is provided.
    if !isnothing(parameters.C)
        check_parameter_observations(parameters)
    end

    # Check if the values are non-negative
    for name in fieldnames(typeof(parameters))
        parameter = getfield(parameters, name)
        
        # If parameter has not been provided, don't check.
        if !isnothing(parameter) && name != :C
            if !ActiveInference.is_non_negative(parameter)
                throw(ArgumentError("All elements must be non-negative in parameter '$(name)'"))
            end
        else
            continue
        end
    end

    # Check if the probability distributions are normalized. Only A, B, D, and E are probability distributions.
    params_check_norm = (;parameters.A, B, D, E)

    for name in fieldnames(typeof(params_check_norm))
        parameter = getfield(params_check_norm, name)
        # If parameter has not been provided, don't check.
        if !isnothing(parameter)
            try 
                check_probability_distribution(parameter)
            catch e
                throw(ArgumentError("The parameter '$name' is not a valid probability distribution."))
            end
        else
            continue
        end
    end

end


"""
Check the settings and making sure they are all formatted and typed correctly.

# Arguments
- `settings::POMDPActiveInferenceSettings`

Throws an error if the settings are not valid:
    - The policy length should be a positive integer.
    - The action selection can only be stochastic or deterministic.
    - The modalities to learn must be a vector of integers or the string "all".
    - The factors to learn must be a vector of integers or the string "all".
    - The number of FPI iterations must be a positive integer.
    - The tolerance for FPI must be a positive float.
"""
function check_settings(settings::POMDPActiveInferenceSettings)

    # Checking that the policy length is a positive integer
    if !is_non_negative(settings.policy_length)
        throw(ArgumentError("policy_length must be a positive integer. Got: $(settings.policy_length)"))
    end

    # Making sure only the allowed action selection types are specified.
    allowed_action_selection = ("stochastic", "deterministic")
    if settings.action_selection ∉ allowed_action_selection
        throw(ArgumentError(throw(ArgumentError("action_selection must be one of $(allowed_action_selection). Got: '$(settings.action_selection)'"))))
    end

    # Check if modalities to learn is either a string "all" or a vector of positive integers
    if !(settings.modalities_to_learn == "all" || (settings.modalities_to_learn isa AbstractVector && ActiveInference.is_non_negative(settings.modalities_to_learn)))
        throw(ArgumentError("modalities_to_learn must be a vector of positive integers or the string 'all'. Got: $(settings.modalities_to_learn)"))
    end

    # Check if factors to learn is either a string "all" or a vector of positive integers
    if !(settings.factors_to_learn == "all" || (settings.factors_to_learn isa AbstractVector && ActiveInference.is_non_negative(settings.factors_to_learn)))
        throw(ArgumentError("factors_to_learn must be a vector of positive integers or the string 'all'. Got: $(settings.factors_to_learn)"))
    end

    # Check if the FPI_n_iter is positive
    if !is_non_negative(settings.FPI_n_iter)
        throw(ArgumentError("FPI_n_iter must be a positive integer. Got: $(settings.FPI_n_iter)"))
    end

    # Check if the FPI_tol is positive
    if !is_non_negative(settings.FPI_tol)
        throw(ArgumentError("FPI_tol must be a positive float. Got: $(settings.FPI_tol)"))
    end

end

"""
Function that checks if the information provided in the settings and parameters are compatible

# Arguments
- `settings::POMDPActiveInferenceSettings`
- `parameters::POMDPActiveInferenceParameters`

Throws an error if the settings and parameters are not compatible:
    - The policy length should be corresponding to the E-parameters if provided.
    - The modalities_to_learn does not include modalities not specified in the A or C parameters.
    - The factors_to_learn does not include factors not specified in the A, B, or D parameters.
"""
function check_settings_and_parameters(settings::POMDPActiveInferenceSettings, parameters::POMDPActiveInferenceParameters)

    # If E is provided, infer the policy length of E and check if compatible with settings policy_length.
    if !isnothing(parameters.E)

        # Making sure this is compatible with both B and its prior
        B_or_pB = isnothing(parameters.B) ? parameters.pB : parameters.B

        # Extracting n_controls
        n_controls = [size(B_or_pB[factor], 3) for factor in eachindex(B_or_pB)]
        policy_length = log(length(parameters.E))/log(prod(n_controls))
        policy_length_int = Int64(round(policy_length, digits = 2))

        # Comparing the extracted policy length with the settings policy length
        if settings.policy_length != policy_length_int
            throw(ArgumentError("The policy length must be equal to the number of E-parameters. Got: settings.policy_length = $(settings.policy_length) and policy length of E = $(policy_length_int)"))
        end
    end 

    # Check if the modalities_to_learn does not include modalities not specified in the A parameters.
    if settings.modalities_to_learn != "all"

        # Making sure this is compatible with both A and its prior
        A_or_pA, name_A_pA = isnothing(parameters.A) ? (parameters.pA, "pA") : (parameters.A, "A")

        # Extracting n_modalities as a vector from both A and C
        modalities_A = Vector(1:1:length(A_or_pA))

        # Check if the modalities_to_learn does not include modalities not specified in the A or C parameters.
        for modality in settings.modalities_to_learn
            if modality ∉ modalities_A
                throw(ArgumentError("The setting modalities_to_learn include a modelity $(modality) which is not included in the $name_A_pA parameters which have modalities: $(modalities_A)"))
            end
        end

    end

    # Check if the factors_to_learn does not include factors not specified in the A, B, or D parameters.
    if settings.factors_to_learn != "all"

        # Making sure this is compatible with both A, B, and D
        A_or_pA, name_A_pA = isnothing(parameters.A) ? (parameters.pA, "pA") : (parameters.A, "A")
        B_or_pB, name_B_pB  = isnothing(parameters.B) ? (parameters.pB, "pB") : (parameters.B, "B")

        # Extracting n_factors as a vector from both A, B, and D
        n_factors_A = length(size(A_or_pA[1])) - 1
        factors_A = Vector(1:1:n_factors_A)
        factors_B = Vector(1:1:length(B_or_pB))

        # Check if the factors_to_learn does not include factors not specified in the A, B, or D parameters.
        for factor in settings.factors_to_learn
            if factor ∉ factors_A && factor ∉ factors_B
                throw(ArgumentError("The setting factors_to_learn include a factor $(factor) which is not included in the $(name_A_pA) and $name_B_pB parameters which have factors: $(factors_B)"))
            end
        end

    end

end


"""
Infer generative model parameters that are not provided.

# Arguments
- `parameters::POMDPActiveInferenceParameters`

If parameters C, D, or E are not provided, they are inferred from the provided parameters pA or A and pB or B.
"""
function infer_missing_parameters(parameters::POMDPActiveInferenceParameters, settings::POMDPActiveInferenceSettings, verbose::Bool = true)

    # If pA is provided, we create A based on pA
    if isnothing(parameters.A)
        parameters.A = normalize_arrays(deepcopy(parameters.pA))
    end

    # If pB is provided, we create B based on pB
    if isnothing(parameters.B)
        parameters.B = normalize_arrays(deepcopy(parameters.pB))
    end

    # If C is not provided, we create C based on the number of observations
    if isnothing(parameters.C)

        # Extracting n_observations
        n_observations = [size(A, 1) for A in parameters.A]

        # Creating C with zero vectors
        parameters.C = [zeros(observation_dimension) for observation_dimension in n_observations]

        if verbose
            @warn "No C-vector provided, no prior preferences will be used."
        end
    end
    
    # If D is not provided, we create either based on pD if provided. Otherwise, we create D based on the number of states
    if isnothing(parameters.D) && isnothing(parameters.pD)
        
        # Extracting n_states
        n_states = [size(B, 1) for B in parameters.B]

        # Uniform D vectors
        parameters.D = [fill(1.0 / state_dimension, state_dimension) for state_dimension in n_states]

        if verbose
            @warn "No D-vector provided, uniform priors over states will be used."
        end

    elseif !isnothing(parameters.pD)
        parameters.D = normalize_arrays(deepcopy(parameters.pD))
    end

    if isnothing(parameters.E)
        # Extracting n_controls and calculating the number of policies
        B_or_pB = isnothing(parameters.B) ? parameters.pB : parameters.B
        n_controls = [size(B_or_pB[factor], 3) for factor in eachindex(B_or_pB)]  
        n_policies = prod(n_controls) ^ settings.policy_length

        # Uniform E vector
        parameters.E = fill(1.0 / n_policies, n_policies)

        if verbose == true
            @warn "No E-vector provided, uniform prior over policies will be used."
        end
    end

end

"""
Function to check if the statefactor dimensions of the parameters are consistent.
"""
function check_parameter_states(parameters::POMDPActiveInferenceParameters)

    # Check the number of states in A/pA, B/pB
    A_or_pA, name_A_pA = isnothing(parameters.A) ? (parameters.pA, "pA") : (parameters.A, "A")
    B_or_pB, name_B_pB = isnothing(parameters.B) ? (parameters.pB, "pB") : (parameters.B, "B")
    
    A_or_pA_states = [size(A_or_pA[1], factor + 1) for factor in 1:length(size(A_or_pA[1])[2:end])]
    B_or_pB_states = [size(B_or_pB[factor], 1) for factor in eachindex(B_or_pB)]
    
    # Check whether to include D or pD
    if !isnothing(parameters.D) || !isnothing(parameters.pD)
        D_or_pD, name_D_pD = isnothing(parameters.D) ? (parameters.pD, "pD") : (parameters.D, "D")
        D_or_pD_states = [size(D_or_pD[factor], 1) for factor in eachindex(D_or_pD)]
    
        # Check consistency between A/pA, B/pB, and D/pD
        if A_or_pA_states != B_or_pB_states || B_or_pB_states != D_or_pD_states
            throw(ArgumentError("""
            The number of states in each factor are different in $name_A_pA, $name_B_pB, and $name_D_pD.
    
            States in $name_A_pA: $A_or_pA_states
            States in $name_B_pB: $B_or_pB_states
            States in $name_D_pD: $D_or_pD_states
            """))
        end
    else
        # Check consistency only between A/pA and B/pB if D/pD is not provided
        if A_or_pA_states != B_or_pB_states
            throw(ArgumentError("""
            The number of states in each factor are different in $name_A_pA and $name_B_pB.
    
            States in $name_A_pA: $A_or_pA_states
            States in $name_B_pB: $B_or_pB_states
            """))
        end
    end
end

"""
Function to check if the number of observationmodalities in the parameters are consistent.
"""
function check_parameter_observations(parameters::POMDPActiveInferenceParameters)

    # Check the number of observations in A/pA and C
    A_or_pA, name_A_pA = isnothing(parameters.A) ? (parameters.pA, "pA") : (parameters.A, "A")
    A_or_pA_observations = [size(A_or_pA[modality], 1) for modality in eachindex(A_or_pA)]
    C_observations = [size(parameters.C[modality], 1) for modality in eachindex(parameters.C)]

    # Throw an error if the number of observations are different
    if A_or_pA_observations != C_observations
        throw(ArgumentError("\n\nThe number of observations are different in $name_A_pA and C \nNumber of observations in parameters: \n\n$name_A_pA: $A_or_pA_observations \nC: $C_observations \n"))
    end

end

"""
Function infers missing settings based on the provided parameters. This function primarily handles hidden fields in the settings struct.

# Arguments
- `settings::POMDPActiveInferenceSettings`
- `parameters::POMDPActiveInferenceParameters`

Returns the updated settings struct.
    - Computes and stores the _n_controls and _control_fac_idx fields.

"""
function infer_missing_settings(settings::POMDPActiveInferenceSettings, parameters::POMDPActiveInferenceParameters)

    # Extracting the number of states and observations from the parameters
    n_states = [size(B, 1) for B in parameters.B]
    n_observations = [size(A, 1) for A in parameters.A]

    # Infer the number of controls in each factor based on B
    _n_controls = [size(parameters.B[factor], 3) for factor in eachindex(parameters.B)]
    _control_fac_idx = [f for f in eachindex(_n_controls) if _n_controls[f] > 1]

    # Updating the settings struct with the inferred fields
    settings._n_states = n_states
    settings._n_observations = n_observations
    settings._n_controls = _n_controls
    settings._control_fac_idx = _control_fac_idx
    settings.policies = construct_policies(settings)

end

"""
    construct_states_struct(parameters::POMDPActiveInferenceParameters, settings::POMDPActiveInferenceSettings)

# Arguments
- `parameters::POMDPActiveInferenceParameters`
- `settings::POMDPActiveInferenceSettings`

Returns an instance of the states struct with default initial values:
    - policies
    - qs_current
    - obs_current
    - prior
    - q_pi
    - G
    - action
    - SAPE
    - bayesian_model_averages
"""
function construct_states_struct(parameters::POMDPActiveInferenceParameters, settings::POMDPActiveInferenceSettings)

    qs_current = [fill(1.0 / state_dimension, state_dimension) for state_dimension in settings._n_states]
    obs_current = zeros(Int, length(settings._n_observations))
    prior = parameters.D
    q_pi = ones(length(settings.policies)) / length(settings.policies)
    G = zeros(length(settings.policies))
    action = Int[]
    SAPE = missing
    bayesian_model_averages = missing

    return POMDPActiveInferenceStates(qs_current, obs_current, prior, q_pi, G, action, SAPE, bayesian_model_averages)
end

"""
    construct_history_struct(states_struct::POMDPActiveInferenceStates)

# Arguments
- `states_struct::POMDPActiveInferenceStates`

Returns an instance of the history struct with value from states_struct and nothing for SAPE and bayesian_model_averages:
    - qs_current
    - obs_current
    - prior
    - q_pi
    - G
    - action
    - SAPE
    - bayesian_model_averages
"""
function construct_history_struct(states_struct::POMDPActiveInferenceStates)

    qs_current = [states_struct.qs_current]
    obs_current = [states_struct.obs_current]
    prior = [states_struct.prior]
    q_pi = [states_struct.q_pi]
    G = [states_struct.G]
    action = [states_struct.action]
    SAPE = missing
    bayesian_model_averages = missing

    return POMDPActiveInferenceHistory(qs_current, obs_current, prior, q_pi, G, action, SAPE, bayesian_model_averages)
end


"""
    construct_policies(settings::POMDPActiveInferenceSettings)

Construct policies based on the number of states, controls, policy length, and indices of controllable state factors.

# Arguments
- `settings::POMDPActiveInferenceSettings`

"""
function construct_policies(settings::POMDPActiveInferenceSettings)

    # Create a vector of possible actions for each time step
    x = repeat(settings._n_controls, settings.policy_length)

    # Generate all combinations of actions across all time steps
    policies = collect(Iterators.product([1:i for i in x]...))

    # Initialize an empty vector to store transformed policies
    transformed_policies = Vector{Matrix{Int64}}()

    for policy_tuple in policies
        # Convert tuple into a vector
        policy_vector = collect(policy_tuple)
        
        # Reshape the policy vector into a matrix and transpose it
        policy_matrix = reshape(policy_vector, (length(policy_vector) ÷ settings.policy_length, settings.policy_length))'
        
        # Push the reshaped matrix to the vector of transformed policies
        push!(transformed_policies, policy_matrix)
    end

    return transformed_policies
end
