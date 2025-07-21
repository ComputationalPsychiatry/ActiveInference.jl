# @infiltrate; @assert false



# Create ActiveInference Agent 
function create_agent(model::NamedTuple, settings::NamedTuple; parameters=missing)
    #=
    Before initializing the agent, we preform validation tests, fill in missing values, give 
    warnings, check types, create qs and EFE objects, and so on. Formal type checking occurs when we 
    initialize the agent. Prior to that we inform the user if there are any problems.
    
    todo: specify exact types for the objects that we create             
    =#
    
    if ismissing(parameters)
        parameters = get_parameters()
    end

    Validate.validate(model, settings, parameters)
    Validate.complete(model, settings, parameters) 

    state_names = [x.name for x in model.states]
    qss = [x.D for x in model.states]
    qs_current = (; zip(state_names, qss)...)
    qs_prior = deepcopy(qs_current)

    obs_names = [x.name for x in model.obs]
    qos = [zeros(x.A_dims[1]) for x in model.obs]
    qo_current = (; zip(obs_names, qos)...)
    
    q_pi = zeros(Union{Missing, Float64}, model.policies.n_policies) 
    G = zeros(Union{Missing, Float64}, model.policies.n_policies)
    EFE = zeros(Union{Missing, Float64}, (model.policies.n_policies, model.policies.policy_length))
    
    # todo: we don't need to record utility etc. in matrices internally if setting.return_EFE_decompositions=false
    utility = zeros(Union{Missing, Float64}, (model.policies.n_policies, model.policies.policy_length))
    info_gain = zeros(Union{Missing, Float64}, (model.policies.n_policies, model.policies.policy_length))
    risk = zeros(Union{Missing, Float64}, (model.policies.n_policies, model.policies.policy_length))
    ambiguity = zeros(Union{Missing, Float64}, (model.policies.n_policies, model.policies.policy_length))
    
    # todo: these can be nothing if no parameter learning
    info_gain_A = zeros(Union{Missing, Float64}, (model.policies.n_policies, model.policies.policy_length))
    info_gain_B = zeros(Union{Missing, Float64}, (model.policies.n_policies, model.policies.policy_length))
    info_gain_D = zeros(Union{Missing, Float64}, (model.policies.n_policies, model.policies.policy_length))

    last_action = nothing 
    
    # initialize history NamedTuple
    history = nothing
    if settings.save_history
        history = (
            actions = [],
            qs = [],
            prior = [],
            posterior_policies = [],
            EFE = [],
            bayesian_model_averages = [],
            SAPE = [],
        )
    end

    #@infiltrate; @assert false
    return Agent(   model,
                    parameters,
                    settings,
                    last_action,
                    qs_prior,
                    qs_current,
                    qo_current,
                    q_pi,
                    G,
                    utility,
                    info_gain, 
                    risk,
                    ambiguity, 
                    info_gain_A, 
                    info_gain_B,
                    info_gain_D,
    )
end


# --------------------------------------------------------------------------------------------------
function get_settings()
    # return settings NamedTuple. Settings are string or other values used to control computations.
    settings = (
        
        # general group
        #use_Float64 = true,  # todo: Float32 not yet implemented
        save_history = false,

        # EFE calculation group
        use_utility = true,    
        use_states_info_gain = true,
        use_param_info_gain = true,  # for parameter learning
        FPI_num_iter = 10,
        FPI_dF_tol = 0.001,
        
        # policy inference group
        policy_inference_method = [:standard, :sophisticated, :inductive][2],
        graph = [:explicit, :implicit, :none][1],
        EFE_over = [:policies, :actions][1],
        graph_postprocessing_method = [:G_prob, :G_prob_q_pi][1],
        early_stop = false,  # if true, user must supply earlystop_tests 
        policy_filtering = false,  # if true, user must supply policy_tests
        EFE_reduction = [:sum, :min_max, :custom][1],  # if early_stop=true, missing values might occur. If :Custom, user must supply EFE_reduction function.
        return_EFE_decompositions = true,  # todo: allow for not returning utility, info gain, etc. matrices
        SI_observation_prune_threshold = 1/16,  
        SI_policy_prune_threshold = 1/16,
        SI_prune_penalty = 512,

        # action group
        action_selection = :stochastic,
        
        # stdout group
        verbose = false,
        warnings = false,
        #logging = false,  #todo: not yet implemented, and need filename  
    )

    return settings
end


# --------------------------------------------------------------------------------------------------
function get_parameters()
    # return parameters NamedTuple. Parameters are scalar values used in calculations.
    parameters = (
        gamma = 16.0,
        alpha = 16.0,
        lr_pA = 1.0,
        fr_pA = 1.0,
        lr_pB = 1.0,
        fr_pB = 1.0,
        lr_pD = 1.0,
        fr_pD = 1.0,  
    )

    return parameters
end


""" Update the agents's beliefs over states """
function infer_states!(agent::Agent, obs::NamedTuple{<:Any, <:NTuple{N, Int64} where {N}}) 
    
    # todo: do similar consistency checks on all incoming objects from user (obs, actions, etc.) 
    # consistency test
    @assert keys(agent.model.obs) == keys(obs)

    if !isnothing(agent.last_action)
        #=
        An action has been taken, and new obs is available, but the agent hasn't processed the obs
        yet. Here we calculate qs_prior, beliefs about states resulting from the action. Note that 
        this very calculation has already been done when the agent calculated q_pi over all actions
        in the last simulation step. But the value of qs_pi for the selected action was not saved. 
        So here we calculate it again. The prior will be used later, in update_posterior_states, which
        uses the new observation.
        =#  
        @infiltrate; @assert false
        
        int_action = round.(Int, agent.action)
        agent.prior = Inference.get_expected_states(
            agent.qs_current, 
            agent.B, 
            reshape(int_action, 1, length(int_action)),  # single policy, e.g., [[3,1,1]]
            agent.metamodel,
        )[1]
    end

    # Update posterior over states
    qs_current = Inference.update_posterior_states(agent, obs)  

    # @set qs in model
    @infiltrate; @assert false
    
    # Adding the obs to the agent struct
    agent.obs_current = obs

    # Push changes to agent's history
    push!(agent.states["prior"], agent.prior)
    push!(agent.states["posterior_states"], agent.qs_current)

    return agent.qs_current
end


""" Update the agents's beliefs over policies """
function infer_policies!(agent::Agent)
    # Update posterior over policies and expected free energies of policies
    
    @assert agent.graph_postprocessing_method in ["G_prob_method", "G_prob_qpi_method", "marginal_EFE_method"]

    if agent.sophisticated_inference | agent.use_SI_graph_for_standard_inference
        q_pi, G, utility, info_gain, risk, ambiguity = Sophisticated.update_posterior_policies(agent)
        info_gain_B  = nothing
    else    
        q_pi, G, utility, info_gain, risk, ambiguity, info_gain_B = update_posterior_policies(agent)
    end

    #@infiltrate; @assert false
    agent.Q_pi = q_pi
    agent.G = G  
    agent.utility = utility
    agent.info_gain = info_gain
    agent.risk = risk
    agent.ambiguity = ambiguity
    agent.info_gain_B = info_gain_B
    

    # Push changes to agent's history
    push!(agent.states["posterior_policies"], copy(agent.Q_pi))
    push!(agent.states["expected_free_energies"], copy(agent.G))

    return q_pi
end


""" Sample action from the beliefs over policies """
function sample_action!(agent::Agent)
    action = sample_action(
        agent.Q_pi, 
        agent.policies, 
        agent.num_controls; 
        action_selection=agent.action_selection, 
        alpha=agent.alpha,
        metamodel = agent.metamodel
    )

    agent.action = action 

    # Push action to agent's history
    push!(agent.states["action"], copy(agent.action))


    return action
end

""" Update A-matrix """
function update_A!(agent::Agent)
    @infiltrate; @assert false
    qA = update_obs_likelihood_dirichlet(agent.pA, agent.A, agent.obs_current, agent.qs_current, lr = agent.lr_pA, fr = agent.fr_pA, modalities = agent.modalities_to_learn)
    
    agent.pA = deepcopy(qA)
    agent.A = deepcopy(normalize_arrays(qA))

    return qA
end

""" Update B-matrix """
function update_B!(agent::Agent)
    #@infiltrate; @assert false
    if length(agent.states["posterior_states"]) > 1

        qs_prev = agent.states["posterior_states"][end-1]

        # todo: why not just pass agent for most of these?
        qB = update_state_likelihood_dirichlet(
                                agent.pB, 
                                agent.B, 
                                agent.action, 
                                agent.qs_current, 
                                qs_prev, 
                                agent.metamodel,
                                lr = agent.lr_pB, 
                                fr = agent.fr_pB, 
                                factors_to_learn = agent.factors_to_learn,  # either "all" or list of symbols?
                            )

        agent.pB = deepcopy(qB)
        agent.B = deepcopy(normalize_arrays(qB))
    else
        qB = nothing
    end

    return qB
end


""" Update D-matrix """
function update_D!(agent::Agent)
    @infiltrate; @assert false
    if length(get_history(agent, "posterior_states")) == 1

        qs_t1 = get_history(agent, "posterior_states")[end]
        qD = update_state_prior_dirichlet(agent.pD, qs_t1; lr = agent.lr_pD, fr = agent.fr_pD, factors = agent.factors_to_learn)

        agent.pD = deepcopy(qD)
        agent.D = deepcopy(normalize_arrays(qD))
    else
        qD = nothing
    end
    return qD
end

""" General Learning Update Function """

function update_parameters!(agent::Agent)

    if agent.pA != nothing
        update_A!(agent)
    end

    if agent.pB != nothing
        update_B!(agent)
    end

    if agent.pD != nothing
        update_D!(agent)
    end
    
end

""" Get the history of the agent """



