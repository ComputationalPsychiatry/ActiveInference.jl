module Validate

import Setfield: @set

using Format
using Infiltrator
using Revise


import ActiveInference.ActiveInferenceFactorized as AI 





# @infiltrate; @assert false



# --------------------------------------------------------------------------------------------------
# test model and settings for correctness
function validate(model, settings, parameters)
    
    #@infiltrate; @assert false
    
    #=
    todo: 
        - these should all be organized by states, obs, actions, policies etc, 
        - use only one iterator over model for each kind
        - include tests for the combination of policy inference settings
        - test that the model, settings, and parameters match their expected templates
        - make sure any `nothing` in the model really can be a `nothing`
        - make sure user-supplied policy tests exist, if settings indicate their use
        - run any user-supplied policy tests on simple data, to make sure they work as expected
        - for settings, warn when the combination will ignore some settings
        - warn if some settings ignored because of model specification (e.g., no parameter learning) 
    =#

    

    # Check A matrix
    #@infiltrate; @assert false
    for obs in model.obs
        try
            if !AI.Utils.check_probability_distribution(obs.A)
                error("The A matrix is not a proper probability distribution.")
            end
        catch e
            # Add context and rethrow the error
            error("The A matrix is not a proper probability distribution. Details: $(e)")
        end
    end

    # Check B matrix
    for state in model.states
        try
            if !AI.Utils.check_probability_distribution(state.B)
                error("The B matrix is not a proper probability distribution.")
            end
        catch e
            # Add context and rethrow the error
            error("The B matrix is not a proper probability distribution. Details: $(e)")
        end
    end

    # Check D matrix (if it's not nothing)
    for state in model.states
        try
            if !isnothing(state.D) && !AI.Utils.check_probability_distribution(state.D)
                error("The D matrix is not a proper probability distribution.")
            end
        catch e
            # Add context and rethrow the error
            error("The D matrix is not a proper probability distribution. Details: $(e)")
        end
    end

    try
        if isnothing(model.policies.policy_iterator) || ismissing(model.policies.policy_iterator) 
            error("The user must supply a policy_iterator.")
        end
    catch e
        # Add context and rethrow the error
        error("The user must supply a policy_iterator")
    end

    try
        if isnothing(model.policies.action_iterator) || ismissing(model.policies.action_iterator)
            error("The user must supply an action_iterator.")
        end
    catch e
        # Add context and rethrow the error
        error("The user must supply an action_iterator")
    end

    try
        if settings.SI_use_pymdp_methods
            # these are the settings that will approx replicate SI in pymdp
            @assert settings.policy_inference_method == :sophisticated
            @assert settings.graph == :explicit
            @assert settings.EFE_over == :actions
            @assert settings.graph_postprocessing_method == :G_prob_q_pi
            @assert settings.EFE_reduction == :sum
        end
    catch e
        s = "To appox replicate pymdp methods, user must choose settings :sophisticated, :explicit "
        s *= format(":actions, :G_prob_q_pi, and :sum. Error is: {}", e)
        error(s)
    end
    

    if settings.EFE_over == :actions && settings.policy_inference_method == :standard && settings.graph == :none
        #@assert false "not yet implemented"  # todo: implement :actions and :standard - maybe just sum over actions at end of infer policies?
    end

    if settings.policy_inference_method == :inductive
        @assert false "not yet implemented"  # todo: implement inductive inference
    end

    #=
    todo:  all these tests needs to be cleaned up, added to, and reorganized

    # Check if parameters are provided or use defaults
    if isnothing(parameters)
        if warn == true
            @warn "No parameters provided, default parameters will be used."
        end
    end

    # Throw warning if no D-vector is provided. 
    for obs in model.obs
        if warn == true && isnothing(obs.C)
            @warn format("No C-vector provided for {}, uninformative prior preferences will be used.", obs.name)
        end 
    end
    
    # Throw warning if no D-vector is provided. 
    if warn == true && isnothing(D)
        @warn "No D-vector provided, a uniform distribution will be used."
    end 
    

    # Throw warning if no E-vector is provided. 
    if warn == true && isnothing(E)
        @warn "No E-vector provided, a uniform distribution will be used."
    end           
    
    # Check if settings are provided or use defaults
    if isnothing(settings)

        if warn == true
            @warn "No settings provided, default settings will be used."
        end

    if isnothing(model.policies.E)


    if !ismissing length(E) != metamodel.number_policies
        error("Length of E-vector must match the number of policies.")
    end

    
    #Print out agent settings
    if warn == true
        settings_summary = 
        """
        Agent Agent initialized successfully with the following settings and parameters:
        - Gamma (γ): $(agent.gamma)
        - Alpha (α): $(agent.alpha)
        - Policy Length: $(agent.policy_len)
        - Number of Controls: $(agent.num_controls)
        - Controllable Factors Indices: $(agent.control_fac_idx)
        - Use Utility: $(agent.use_utility)
        - Use States Information Gain: $(agent.use_states_info_gain)
        - Use Parameter Information Gain: $(agent.use_param_info_gain)
        - Action Selection: $(agent.action_selection)
        - Modalities to Learn = $(agent.modalities_to_learn)
        - Factors to Learn = $(agent.factors_to_learn)
        """
        println(settings_summary)
    end

    =#

end


# --------------------------------------------------------------------------------------------------
function complete(model, settings, parameters)
    #=
    todo:
        - fill in any missings in template that the user has not filled in
    =#
    
    
    # if E-vector is not provided
    if ismissing(model.policies.E)
        # use uninformative prior on policies
        if settings.EFE_over == :policies
            model = @set model.policies.E = ones(Real, model.policies.n_policies) / model.policies.n_policies
        elseif settings.EFE_over == :actions
            n_actions = length(collect(model.policies.action_iterator))
            model = @set model.policies.E = ones(Real, n_actions) / n_actions
        end
    end

    return model

end


end  # --- module