# in dev package home, Pkg.activate("./") to add packages, then Pkg.resolve() in EcoNet


""" -------- AIF Mutable Struct -------- """



using Format
using Infiltrator
using Revise

#show(stdout, "text/plain", x)
# @infiltrate; @assert false



mutable struct AIF
    #A::Vector{Array{T, N}} where {T <: Real, N} # A-matrix
    #B::Vector{Array{T, N}} where {T <: Real, N} # B-matrix
    A::Union{Vector{Array{T}} where {T <: Real}, Vector{Array{T, N}} where {T <: Real, N}}  # A-matrix
    B::Union{Vector{Array{T}} where {T <: Real}, Vector{Array{T, N}} where {T <: Real, N}} # B-matrix
    
    
    C::Vector{Array{Real}} # C-vectors
    D::Vector{Vector{Real}} # D-vectors
    E::Vector{T} where T <: Real       # E-vector (Habits)
    #pA::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} # Dirichlet priors for A-matrix
    #pB::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} # Dirichlet priors for B-matrix
    pA::Union{Vector{Array{T}}, Nothing} where {T <: Real} # Dirichlet priors for A-matrix
    pB::Union{Vector{Array{T}}, Nothing} where {T <: Real} # Dirichlet priors for B-matrix
    
    
    pD::Union{Vector{Array{Real}}, Nothing} # Dirichlet priors for D-vector
    lr_pA::Real # pA Learning Parameter
    fr_pA::Real # pA Forgetting Parameter,  1.0 for no forgetting
    lr_pB::Real # pB learning Parameter
    fr_pB::Real # pB Forgetting Parameter
    lr_pD::Real # pD Learning parameter
    fr_pD::Real # PD Forgetting parameter
    modalities_to_learn::Union{String, Vector{Int64}} # Modalities can be eithe "all" or "# modality"
    factors_to_learn::Union{String, Vector{Int64}} # Modalities can be either "all" or "# factor"
    gamma::Real # Gamma parameter
    alpha::Real # Alpha parameter
    policies
    num_controls::Array{Int,1} # Number of actions per factor
    control_fac_idx::Array{Int,1} # Indices of controllable factors
    policy_len::Int  # Policy length
    qs_current::Vector{Vector{T}} where T <: Real # Current beliefs about states
    obs_current::Vector{T} where T <: Real # Current observation
    prior::Vector{Vector{T}} where T <: Real # Prior beliefs about states
    Q_pi::Vector{T} where T <:Real # Posterior beliefs over policies
    G::Vector{T} where T <:Real  # Expected free energy of policies
    utility::Matrix{Union{Missing, Float64}} 
    info_gain::Matrix{Union{Missing, Float64}} 
    risk::Matrix{Union{Missing, Float64}} 
    ambiguity::Matrix{Union{Missing, Float64}} 
    action::Vector{Int} # Last action
    use_utility::Bool # Utility Boolean Flag
    use_states_info_gain::Bool # States Information Gain Boolean Flag
    use_param_info_gain::Bool # Include the novelty value in the learning parameters
    action_selection::String # Action selection: can be either "deterministic" or "stochastic"
    FPI_num_iter::Int # Number of iterations stopping condition in the FPI algorithm
    FPI_dF_tol::Float64 # Free energy difference stopping condition in the FPI algorithm
    states::Dict{String,Array{Any,1}} # States Dictionary
    parameters::Dict{String,Real} # Parameters Dictionary
    settings::Dict{String,Any} # Settings Dictionary
    save_history::Bool # Save history boolean flag
    sophisticated_inference::Bool
    horizon::Int64
    metamodel
end

# Create ActiveInference Agent 
function create_aif(A, B;
                    C = nothing,
                    D = nothing,
                    E = nothing,
                    pA = nothing, 
                    pB = nothing, 
                    pD = nothing, 
                    lr_pA = 1.0, 
                    fr_pA = 1.0, 
                    lr_pB = 1.0, 
                    fr_pB = 1.0, 
                    lr_pD = 1.0, 
                    fr_pD = 1.0, 
                    modalities_to_learn = "all", 
                    factors_to_learn = "all", 
                    gamma=1.0, 
                    alpha=1.0, 
                    policy_len=1, 
                    num_controls=nothing, 
                    control_fac_idx=nothing, 
                    use_utility=true, 
                    use_states_info_gain=true, 
                    use_param_info_gain = false, 
                    action_selection="stochastic",
                    FPI_num_iter=10,
                    FPI_dF_tol=0.001,
                    save_history=true,
                    sophisticated_inference=false,
                    horizon = 1,
                    metamodel = metamodel,
    )

    num_states = [size(B[f], 1) for f in eachindex(B)]
    num_obs = [size(A[f], 1) for f in eachindex(A)]

    # If C-vectors are not provided
    if isnothing(C)
        C = create_matrix_templates(num_obs, "zeros")
    end

    # If D-vectors are not provided
    if isnothing(D)
        D = create_matrix_templates(num_states)
    end

    # if num_controls are not given, they are inferred from the B matrix
    if isnothing(num_controls)
        num_controls = [size(B[f], 3) for f in eachindex(B)]  
    end

    # Determine which factors are controllable
    if isnothing(control_fac_idx)
        control_fac_idx = [f for f in eachindex(num_controls) if num_controls[f] > 1]
    end

    if false
        # original
        policies = construct_policies(
            num_states, 
            n_controls=num_controls, 
            policy_length=policy_len, 
            controllable_factors_indices=control_fac_idx
        )
    end

    policies = metamodel.policies

    # if E-vector is not provided
    if isnothing(E)
        #E = ones(Real, length(policies)) / length(policies)
        E = ones(Real, policies.number_policies) / policies.number_policies
    end

    # Throw error if the E-vector does not match the length of policies
    #if length(E) != length(policies)
    if length(E) != policies.number_policies
        error("Length of E-vector must match the number of policies.")
    end

    qs_current = create_matrix_templates(num_states)
    obs_current = zeros(Int, length(num_obs))
    prior = D

    #Q_pi = ones(length(policies)) / length(policies)  
    #G = zeros(length(policies))
    #utility = zeros(length(policies), policies[1].size[1])
    #info_gain = zeros(length(policies), policies[1].size[1])
    
    Q_pi = ones(policies.number_policies) / policies.number_policies  
    G = zeros(policies.number_policies)
    
    utility = Matrix{Float64}(undef, 0, 0)  #zeros(policies.number_policies, policies[1].size[1])
    info_gain = Matrix{Float64}(undef, 0, 0)  #zeros(length(policies), policies[1].size[1])
    risk = Matrix{Float64}(undef, 0, 0)  #zeros(policies.number_policies, policies[1].size[1])
    ambiguity = Matrix{Float64}(undef, 0, 0)  #zeros(length(policies), policies[1].size[1])
    
    action = Int[]
    
    # initialize states dictionary
    states = Dict(
        "action" => Vector{Real}[],
        "posterior_states" => Vector{Any}[],
        "prior" => Vector{Any}[],
        "posterior_policies" => Vector{Any}[],
        "expected_free_energies" => Vector{Any}[],
        #"policies" => Any,
        "bayesian_model_averages" => Vector{Vector{<:Real}}[],
        "SAPE" => Vector{<:Real}[]
    )

    # initialize parameters dictionary
    parameters = Dict(
        "gamma" => gamma,
        "alpha" => alpha,
        "lr_pA" => lr_pA,
        "fr_pA" => fr_pA,
        "lr_pB" => lr_pB,
        "fr_pB" => fr_pB,
        "lr_pD" => lr_pD,
        "fr_pD" => fr_pD
    )

    # initialize settings dictionary
    settings = Dict(
        "policy_len" => policy_len,
        "num_controls" => num_controls,
        "control_fac_idx" => control_fac_idx,
        "use_utility" => use_utility,
        "use_states_info_gain" => use_states_info_gain,
        "use_param_info_gain" => use_param_info_gain,
        "action_selection" => action_selection,
        "modalities_to_learn" => modalities_to_learn,
        "factors_to_learn" => factors_to_learn,
        "FPI_num_iter" => FPI_num_iter,
        "FPI_dF_tol" => FPI_dF_tol
    )

    #@infiltrate; @assert false
    return AIF( A,
                B,
                C, 
                D, 
                E,
                pA, 
                pB, 
                pD,
                lr_pA, 
                fr_pA, 
                lr_pB, 
                fr_pB, 
                lr_pD, 
                fr_pD, 
                modalities_to_learn, 
                factors_to_learn, 
                gamma, 
                alpha, 
                policies, 
                num_controls, 
                control_fac_idx, 
                policy_len, 
                qs_current, 
                obs_current,
                prior, 
                Q_pi, 
                G, 
                utility,
                info_gain,
                risk,
                ambiguity,
                action, 
                use_utility,
                use_states_info_gain, 
                use_param_info_gain, 
                action_selection, 
                FPI_num_iter, 
                FPI_dF_tol,
                states,
                parameters, 
                settings, 
                save_history,
                sophisticated_inference,
                horizon,
                metamodel,
                )
end

"""
Initialize Active Inference Agent
function init_aif(
        A,
        B;
        C=nothing,
        D=nothing,
        E = nothing,
        pA = nothing,
        pB = nothing, 
        pD = nothing,
        parameters::Union{Nothing, Dict{String,Real}} = nothing,
        settings::Union{Nothing, Dict} = nothing,
        save_history::Bool = true)

# Arguments
- 'A': Relationship between hidden states and observations.
- 'B': Transition probabilities.
- 'C = nothing': Prior preferences over observations.
- 'D = nothing': Prior over initial hidden states.
- 'E = nothing': Prior over policies. (habits)
- 'pA = nothing':
- 'pB = nothing':
- 'pD = nothing':
- 'parameters::Union{Nothing, Dict{String,Real}} = nothing':
- 'settings::Union{Nothing, Dict} = nothing':
- 'settings::Union{Nothing, Dict} = nothing':

"""
function init_aif(A, B; C=nothing, D=nothing, E=nothing, pA=nothing, pB=nothing, pD=nothing,
                  parameters::Union{Nothing, Dict{String, T}} where T<:Real = nothing,
                  settings::Union{Nothing, Dict} = nothing,
                  save_history::Bool = true, 
                  sophisticated_inference = false,
                  horizon = 1,
                  verbose::Bool = true,
                  metamodel = nothing,
    )

    # Catch error if A, B or D is not a proper probability distribution  
    # Check A matrix
    try
        if !check_probability_distribution(A)
            error("The A matrix is not a proper probability distribution.")
        end
    catch e
        # Add context and rethrow the error
        error("The A matrix is not a proper probability distribution. Details: $(e)")
    end

    # Check B matrix
    try
        if !check_probability_distribution(B)
            error("The B matrix is not a proper probability distribution.")
        end
    catch e
        # Add context and rethrow the error
        error("The B matrix is not a proper probability distribution. Details: $(e)")
    end

    # Check D matrix (if it's not nothing)
    try
        if !isnothing(D) && !check_probability_distribution(D)
            error("The D matrix is not a proper probability distribution.")
        end
    catch e
        # Add context and rethrow the error
        error("The D matrix is not a proper probability distribution. Details: $(e)")
    end

    # Throw warning if no D-vector is provided. 
    if verbose == true && isnothing(C)
        @warn "No C-vector provided, no prior preferences will be used."
    end 

    # Throw warning if no D-vector is provided. 
    if verbose == true && isnothing(D)
        @warn "No D-vector provided, a uniform distribution will be used."
    end 

    # Throw warning if no E-vector is provided. 
    if verbose == true && isnothing(E)
        @warn "No E-vector provided, a uniform distribution will be used."
    end           
    
    # Check if settings are provided or use defaults
    if isnothing(settings)

        if verbose == true
            @warn "No settings provided, default settings will be used."
        end

        settings = Dict(
            "policy_len" => 1, 
            "num_controls" => nothing, 
            "control_fac_idx" => nothing, 
            "use_utility" => true, 
            "use_states_info_gain" => true, 
            "use_param_info_gain" => false,
            "action_selection" => "stochastic", 
            "modalities_to_learn" => "all",
            "factors_to_learn" => "all",
            "FPI_num_iter" => 10,
            "FPI_dF_tol" => 0.001
        )
    end

    # Check if parameters are provided or use defaults
    if isnothing(parameters)

        if verbose == true
            @warn "No parameters provided, default parameters will be used."
        end
        
        parameters = Dict("gamma" => 16.0,
                          "alpha" => 16.0,
                          "lr_pA" => 1.0,
                          "fr_pA" => 1.0,
                          "lr_pB" => 1.0,
                          "fr_pB" => 1.0,
                          "lr_pD" => 1.0,
                          "fr_pD" => 1.0
                          )
    end

    # Extract parameters and settings from the dictionaries or use defaults
    gamma = get(parameters, "gamma", 16.0)  
    alpha = get(parameters, "alpha", 16.0)
    lr_pA = get(parameters, "lr_pA", 1.0)
    fr_pA = get(parameters, "fr_pA", 1.0)
    lr_pB = get(parameters, "lr_pB", 1.0)
    fr_pB = get(parameters, "fr_pB", 1.0)
    lr_pD = get(parameters, "lr_pD", 1.0)
    fr_pD = get(parameters, "fr_pD", 1.0)

    
    policy_len = get(settings, "policy_len", 1)
    num_controls = get(settings, "num_controls", nothing)
    control_fac_idx = get(settings, "control_fac_idx", nothing)
    use_utility = get(settings, "use_utility", true)
    use_states_info_gain = get(settings, "use_states_info_gain", true)
    use_param_info_gain = get(settings, "use_param_info_gain", false)
    action_selection = get(settings, "action_selection", "stochastic")
    modalities_to_learn = get(settings, "modalities_to_learn", "all" )
    factors_to_learn = get(settings, "factors_to_learn", "all" )
    FPI_num_iter = get(settings, "FPI_num_iter", 10 )
    FPI_dF_tol = get(settings, "FPI_dF_tol", 0.001 )

    # Call create_aif 
    aif = create_aif(A, B,
                    C=C,
                    D=D,
                    E=E,
                    pA=pA,
                    pB=pB,
                    pD=pD,
                    lr_pA = lr_pA, 
                    fr_pA = fr_pA, 
                    lr_pB = lr_pB, 
                    fr_pB = fr_pB, 
                    lr_pD = lr_pD, 
                    fr_pD = fr_pD,
                    modalities_to_learn=modalities_to_learn,
                    factors_to_learn=factors_to_learn,
                    gamma=gamma,
                    alpha=alpha, 
                    policy_len=policy_len,
                    num_controls=num_controls,
                    control_fac_idx=control_fac_idx, 
                    use_utility=use_utility, 
                    use_states_info_gain=use_states_info_gain, 
                    use_param_info_gain=use_param_info_gain,
                    action_selection=action_selection,
                    FPI_num_iter=FPI_num_iter,
                    FPI_dF_tol=FPI_dF_tol,
                    save_history=save_history,
                    sophisticated_inference = sophisticated_inference,
                    horizon = horizon,
                    metamodel=metamodel,
                    )

    #Print out agent settings
    if verbose == true
        settings_summary = 
        """
        AIF Agent initialized successfully with the following settings and parameters:
        - Gamma (γ): $(aif.gamma)
        - Alpha (α): $(aif.alpha)
        - Policy Length: $(aif.policy_len)
        - Number of Controls: $(aif.num_controls)
        - Controllable Factors Indices: $(aif.control_fac_idx)
        - Use Utility: $(aif.use_utility)
        - Use States Information Gain: $(aif.use_states_info_gain)
        - Use Parameter Information Gain: $(aif.use_param_info_gain)
        - Action Selection: $(aif.action_selection)
        - Modalities to Learn = $(aif.modalities_to_learn)
        - Factors to Learn = $(aif.factors_to_learn)
        """
        println(settings_summary)
    end
    
    return aif
end

### Struct related functions ###

"""
    construct_policies(n_states::Vector{T} where T <: Real; n_controls::Union{Vector{T}, Nothing} where T <: Real=nothing, 
                       policy_length::Int=1, controllable_factors_indices::Union{Vector{Int}, Nothing}=nothing)

Construct policies based on the number of states, controls, policy length, and indices of controllable state factors.

# Arguments
- `n_states::Vector{T} where T <: Real`: A vector containing the number of  states for each factor.
- `n_controls::Union{Vector{T}, Nothing} where T <: Real=nothing`: A vector specifying the number of allowable actions for each state factor. 
- `policy_length::Int=1`: The length of policies. (planning horizon)
- `controllable_factors_indices::Union{Vector{Int}, Nothing}=nothing`: A vector of indices identifying which state factors are controllable.

"""
function construct_policies(
    n_states::Vector{T} where T <: Real; 
    n_controls::Union{Vector{T}, Nothing} where T <: Real=nothing, 
    policy_length::Int=1, 
    controllable_factors_indices::Union{Vector{Int}, Nothing}=nothing
    )

    @infiltrate; @assert false,  "\nNot used.\n"
    # Determine the number of state factors
    n_factors = length(n_states)

    # If indices of controllable factors are not given 
    if isnothing(controllable_factors_indices)
        if !isnothing(n_controls)
            # Determine controllable factors based on which factors have more than one control
            controllable_factors_indices = findall(x -> x > 1, n_controls)
        else
            # If no controls are given, assume all factors are controllable
            controllable_factors_indices = 1:n_factors
        end
    end

    # if number of controls is not given, determine it based n_states and controllable_factors_indices
    if isnothing(n_controls)
        n_controls = [in(factor_index, controllable_factors_indices) ? n_states[factor_index] : 1 for factor_index in 1:n_factors]
    end

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

    @infiltrate; @assert false
    return transformed_policies
end



""" Update the agents's beliefs over states """
function infer_states!(
    aif::AIF, 
    obs::Vector{Int64}
    )
    
    if !isempty(aif.action)
        #@infiltrate; @assert false
        int_action = round.(Int, aif.action)
        aif.prior = get_expected_states(
            aif.qs_current, 
            aif.B, 
            reshape(int_action, 1, length(int_action)),  # single policy, e.g., [[3,1,1]]
            aif.metamodel,
        )[1]
    else
        aif.prior = aif.D
    end

    # Update posterior over states
    aif.qs_current = update_posterior_states(
        aif.A, 
        aif.metamodel,
        obs, 
        prior=aif.prior, 
        num_iter=aif.FPI_num_iter, 
        dF_tol=aif.FPI_dF_tol
    )

    # Adding the obs to the agent struct
    aif.obs_current = obs

    # Push changes to agent's history
    push!(aif.states["prior"], aif.prior)
    push!(aif.states["posterior_states"], aif.qs_current)

    return aif.qs_current
end

""" Update the agents's beliefs over policies """
function infer_policies!(aif::AIF)
    # Update posterior over policies and expected free energies of policies
    
    if aif.sophisticated_inference
        q_pi, G, utility, info_gain, risk, ambiguity = Sophisticated.update_posterior_policies(aif)
    else    
        q_pi, G, utility, info_gain, risk, ambiguity = update_posterior_policies(aif)
    end

    #@infiltrate; @assert false
    aif.Q_pi = q_pi
    aif.G = G  
    aif.utility = utility
    aif.info_gain = info_gain
    aif.risk = risk
    aif.ambiguity = ambiguity
    

    # Push changes to agent's history
    push!(aif.states["posterior_policies"], copy(aif.Q_pi))
    push!(aif.states["expected_free_energies"], copy(aif.G))

    return q_pi
end


""" Sample action from the beliefs over policies """
function sample_action!(aif::AIF)
    action = sample_action(
        aif.Q_pi, 
        aif.policies, 
        aif.num_controls; 
        action_selection=aif.action_selection, 
        alpha=aif.alpha,
        metamodel = aif.metamodel
    )

    aif.action = action 

    # Push action to agent's history
    push!(aif.states["action"], copy(aif.action))


    return action
end

""" Update A-matrix """
function update_A!(aif::AIF)
    @infiltrate; @assert false
    qA = update_obs_likelihood_dirichlet(aif.pA, aif.A, aif.obs_current, aif.qs_current, lr = aif.lr_pA, fr = aif.fr_pA, modalities = aif.modalities_to_learn)
    
    aif.pA = deepcopy(qA)
    aif.A = deepcopy(normalize_arrays(qA))

    return qA
end

""" Update B-matrix """
function update_B!(aif::AIF)
    @infiltrate; @assert false
    if length(get_history(aif, "posterior_states")) > 1

        qs_prev = get_history(aif, "posterior_states")[end-1]

        qB = update_state_likelihood_dirichlet(aif.pB, aif.B, aif.action, aif.qs_current, qs_prev, lr = aif.lr_pB, fr = aif.fr_pB, factors = aif.factors_to_learn)

        aif.pB = deepcopy(qB)
        aif.B = deepcopy(normalize_arrays(qB))
    else
        qB = nothing
    end

    return qB
end

""" Update D-matrix """
function update_D!(aif::AIF)
    @infiltrate; @assert false
    if length(get_history(aif, "posterior_states")) == 1

        qs_t1 = get_history(aif, "posterior_states")[end]
        qD = update_state_prior_dirichlet(aif.pD, qs_t1; lr = aif.lr_pD, fr = aif.fr_pD, factors = aif.factors_to_learn)

        aif.pD = deepcopy(qD)
        aif.D = deepcopy(normalize_arrays(qD))
    else
        qD = nothing
    end
    return qD
end

""" General Learning Update Function """

function update_parameters!(aif::AIF)

    if aif.pA != nothing
        update_A!(aif)
    end

    if aif.pB != nothing
        update_B!(aif)
    end

    if aif.pD != nothing
        update_D!(aif)
    end
    
end

""" Get the history of the agent """



mutable struct ObsNode
    qs_next::Union{Nothing, Vector{Vector{Float64}}}
    utility_updated::Union{Missing, Nothing, Float64}
    info_gain_updated::Union{Missing, Nothing, Float64}
    ambiguity_updated::Union{Missing, Nothing, Float64}
    risk_updated::Union{Missing, Nothing, Float64}
    G_updated::Union{Missing, Nothing, Float64}
    q_pi_updated::Union{Missing, Nothing, Float64}
    
    prob::Union{Missing, Nothing, Float64}
    prob_updated::Union{Missing, Nothing, Float64}
    
    subpolicy::Union{Missing, Nothing, Tuple}
    outcome::Union{Missing, Nothing, Tuple}
    level::Int64
    ith_outcome::Int64
end

mutable struct ActionNode
    qs::Union{Nothing, Vector{Vector{Float64}}}
    qs_pi::Union{Nothing, Vector{Vector{Vector{Float64}}}}
    qo_pi::Union{Nothing, Vector{Vector{Vector{Float64}}}}  
    utility::Union{Missing, Nothing, Float64}
    info_gain::Union{Missing, Nothing, Float64}
    ambiguity::Union{Missing, Nothing, Float64}
    risk::Union{Missing, Nothing, Float64}
    G::Union{Missing, Nothing, Float64}
    
    utility_updated::Union{Missing, Nothing, Float64}
    info_gain_updated::Union{Missing, Nothing, Float64}
    ambiguity_updated::Union{Missing, Nothing, Float64}
    risk_updated::Union{Missing, Nothing, Float64}
    G_updated::Union{Missing, Nothing, Float64}
    q_pi_updated::Union{Missing, Nothing, Float64}
    
    action::Union{Missing, Nothing, Int64}
    subpolicy::Union{Missing, Nothing, Tuple}
    outcome::Union{Missing, Nothing, Tuple}
    level::Int64
end


struct EarlyStop
    msg::String 
    
end

struct BadPath
    msg::String 
end


mutable struct GraphEdge  
end


struct Label
    level::Int64
    outcome::Union{Nothing, NTuple{N, Int64} where N}
    action::Union{Nothing, Int64}
    type::String
end
