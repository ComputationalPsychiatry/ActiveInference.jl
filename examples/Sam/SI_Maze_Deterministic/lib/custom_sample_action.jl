function custom_sample_action!(agent)

    action_names = [x.name for x in agent.model.actions]

    model = agent.model
    if agent.settings.EFE_over == :actions
        q_pi = agent.q_pi_actions
        G = agent.G_actions
        policies = model.policies.action_iterator
    else
        q_pi = agent.q_pi_policies
        G = agent.G_policies
        policies = model.policies.policy_iterator
    end
    # sample action
    idx = findall(x -> !ismissing(x), G) 
    if agent.settings.action_selection == :deterministic
        ii = idx[argmax(q_pi[idx])]  # argmax over valid policies
    elseif agent.settings.action_selection == :stochastic
        mnd = Distributions.Multinomial(1, Float64.(q_pi[idx]))
        ii = argmax(rand(mnd))
        ii = idx[ii]
    end

    policy = IterTools.nth(policies, ii)
    action_ids = collect(zip(policy...))[1]
    action = (; zip(action_names, action_ids)...)
    
    agent.last_action = action

    # Push action to agent's history
    push!(agent.history.actions, action)
    return action
end

