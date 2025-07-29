# @infiltrate; @assert false

import Setfield: @set


#using Format
#using Infiltrator
#using Revise

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
    model = Validate.complete(model, settings, parameters) 

    state_names = [x.name for x in model.states]
    qss = [x.D for x in model.states]
    qs_current = (; zip(state_names, qss)...)
    qs_prior = deepcopy(qs_current)

    obs_names = [x.name for x in model.obs]
    qos = [zeros(x.A_dims[1]) for x in model.obs]
    qo_current = (; zip(obs_names, qos)...)

    G_actions = nothing
    
    if (settings.EFE_over == :policies 
        ||
        # for standard inference and no graph, base G_marginal and q_pi on full utility etc. matrix
        settings.EFE_over == :actions && settings.policy_inference_method == :standard && settings.graph == :none
        )
        
        G = zeros(Union{Missing, Float64}, model.policies.n_policies)

        if settings.EFE_over == :actions && settings.policy_inference_method == :standard && settings.graph == :none
            n_actions = length(collect(model.policies.action_iterator))
            q_pi = zeros(Union{Missing, Float64}, n_actions) 
            G_actions = zeros(Union{Missing, Float64}, n_actions)
        else
            q_pi = zeros(Union{Missing, Float64}, model.policies.n_policies) 
        end

        # todo: we don't need to record utility etc. in matrices internally if setting.return_EFE_decompositions=false
        utility = zeros(Union{Missing, Float64}, (model.policies.n_policies, model.policies.policy_length))
        info_gain = zeros(Union{Missing, Float64}, (model.policies.n_policies, model.policies.policy_length))
        risk = zeros(Union{Missing, Float64}, (model.policies.n_policies, model.policies.policy_length))
        ambiguity = zeros(Union{Missing, Float64}, (model.policies.n_policies, model.policies.policy_length))
        
        info_gain_A = nothing
        info_gain_B = nothing
        info_gain_D = nothing

        if !all([isnothing(obs.pA) for obs in model.obs]) 
            info_gain_A = zeros(Union{Missing, Float64}, (model.policies.n_policies, model.policies.policy_length))
        end

        if !all([isnothing(state.pB) for state in model.states]) 
            info_gain_B = zeros(Union{Missing, Float64}, (model.policies.n_policies, model.policies.policy_length))
        end

        if !all([isnothing(state.pD) for state in model.states]) 
            info_gain_D = zeros(Union{Missing, Float64}, (model.policies.n_policies, model.policies.policy_length))
        end
    
    elseif settings.EFE_over == :actions
        n_actions = length(collect(model.policies.action_iterator))
        
        q_pi = zeros(Union{Missing, Float64}, n_actions) 
        G = zeros(Union{Missing, Float64}, n_actions)
        EFE = zeros(Union{Missing, Float64}, n_actions)
        
        # todo: we don't need to record utility etc. in matrices internally if setting.return_EFE_decompositions=false
        utility = zeros(Union{Missing, Float64}, (n_actions, 1))
        info_gain = zeros(Union{Missing, Float64}, (n_actions, 1))
        risk = zeros(Union{Missing, Float64}, (n_actions, 1))
        ambiguity = zeros(Union{Missing, Float64}, (n_actions, 1))
        
        info_gain_A = nothing
        info_gain_B = nothing
        info_gain_D = nothing

        if !all([isnothing(obs.pA) for obs in model.obs]) 
            info_gain_A = zeros(Union{Missing, Float64}, (n_actions, 1))
        end

        if !all([isnothing(state.pB) for state in model.states]) 
            info_gain_B = zeros(Union{Missing, Float64}, (n_actions, 1))
        end

        if !all([isnothing(state.pD) for state in model.states]) 
            info_gain_D = zeros(Union{Missing, Float64}, (n_actions, 1))
        end
    end
    
    last_action = nothing 
    
    # initialize history NamedTuple
    history = (
        qs_current = [],
        actions = [],
    )
    if settings.save_history
        history = (
            qs_current = [],
            qs_prior = [],
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
                    G_actions,
                    utility,
                    info_gain, 
                    risk,
                    ambiguity, 
                    info_gain_A, 
                    info_gain_B,
                    info_gain_D,
                    history,
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
        use_param_info_gain = false,  # for parameter learning
        FPI_num_iter = 10,
        FPI_dF_tol = 0.001,
        
        # policy inference group
        policy_inference_method = [:standard, :sophisticated, :inductive][2],
        graph = [:explicit, :implicit, :none][1],
        EFE_over = [:policies, :actions][1],
        graph_postprocessing_method = [:G_prob, :G_prob_q_pi][1],
        EFE_reduction = [:sum, :min_max, :custom][1],  # if early_stop=true, missing values might occur. If :Custom, user must supply EFE_reduction function.
        return_EFE_decompositions = true,  # todo: allow for not returning utility, info gain, etc. matrices
        SI_observation_prune_threshold = 1/16,  
        SI_policy_prune_threshold = 1/16,
        SI_prune_penalty = 512,  # todo: unused for now
        SI_use_pymdp_methods = false,  # flag to calculate EFE as per pymdp

        # action group
        action_selection = [:stochastic, :deterministic][1],
        
        # stdout group
        verbose = false,
        warnings = false,
        #logging = false,  #todo: not yet implemented, and need filename  
    )

    #=
    SI_use_pymdp_methods:
        - do not prune last layer of graph
        - do not remove upstream nodes of pruned node
        - use only one round of pruning
        - user must still choose :sophisticated, :actions, :G_prob_q_pi for SI to resemble pymdp.jax explicit graph methods
    =#

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
        agent.qs_prior = Inference.get_expected_states(
            agent.qs_current, 
            agent.last_action,
            agent 
        )[1] 
    end

    # Update posterior over states
    qs_current = Inference.update_posterior_states(agent, obs)  

    # @set qs in model
    agent.qs_current = qs_current
    
    # Adding the obs to the agent struct
    #agent.obs_current = obs

    # Push changes to agent's history
    push!(agent.history.qs_current, deepcopy(agent.qs_current))
    if agent.settings.save_history
        push!(agent.history.qs_prior, deepcopy(agent.qs_prior))
    end

    return 
end


""" Update the agents's beliefs over policies """
function infer_policies!(agent::Agent, obs_current::NamedTuple{<:Any, <:NTuple{N, T} where {N, T}})
    # Update posterior over policies and expected free energies of policies
    
    if agent.settings.policy_inference_method == :sophisticated || agent.settings.graph == :explicit
        Sophisticated.update_posterior_policies!(agent, obs_current)
    else    
        Inference.update_posterior_policies!(agent)
    end

    # Push changes to agent's history
    if agent.settings.save_history
        push!(agent.history.posterior_policies, deepcopy(agent.q_pi))
        push!(agent.history.EFE, deepcopy(agent.EFE))
    end

    return 
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
    return 
end


""" Update A-matrix """
function update_A!(agent::Agent)
    @infiltrate; @assert false
    qA = update_obs_likelihood_dirichlet(agent.pA, agent.A, agent.obs_current, agent.qs_current, lr = agent.lr_pA, fr = agent.fr_pA, modalities = agent.modalities_to_learn)
    
    agent.pA = deepcopy(qA)
    agent.A = deepcopy(normalize_arrays(qA))
    return 
end


""" Update B-matrix """
function update_B!(agent::Agent)
    
    if length(agent.history.qs_current) > 1  

        qs_prev = agent.history.qs_current[end-1]
        Learning.update_state_likelihood_dirichlet!(agent, qs_prev)
    end
    #@infiltrate; @assert false
    return
end


""" Update D-matrix """
function update_D!(agent::Agent)
    @infiltrate; @assert false
    if length(get_history(agent, "posterior_states")) == 1

        qs_t1 = get_history(agent, "posterior_states")[end]
        qD = update_state_prior_dirichlet(agent.pD, qs_t1; lr = agent.lr_pD, fr = agent.fr_pD, factors = agent.factors_to_learn)

        agent.pD = deepcopy(qD)
        agent.D = deepcopy(normalize_arrays(qD))
    end
    return 
end


""" General Learning Update Function """

function update_parameters!(agent::Agent)
    
    if !isnothing(agent.info_gain_A)
        update_A!(agent)
    end

    if !isnothing(agent.info_gain_B)
        update_B!(agent)
    end

    if !isnothing(agent.info_gain_D)
        update_D!(agent)
    end

    #@infiltrate; @assert false
end





