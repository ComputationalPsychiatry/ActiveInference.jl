function custom_sample_action!(agent, step_i)

    verbose = true
    action_names = [x.name for x in agent.model.actions]

    model = agent.model
    if agent.settings.EFE_over == :actions
        q_pi = agent.q_pi_actions
        G = agent.G_actions
        policies = model.policies.action_iterator
        println("\nInference is over actions.\n")
    else
        q_pi = agent.q_pi_policies
        G = agent.G_policies
        policies = model.policies.policy_iterator
        println("\nInference is over policies.\n")
    end

    # sample action
    t00 = Dates.time()
    idx = findall(x -> !ismissing(x), G) 
    if agent.settings.action_selection == :deterministic
        ii = idx[argmax(q_pi[idx])]  # argmax over valid policies
    elseif agent.settings.action_selection == :stochastic
        mnd = Distributions.Multinomial(1, CONFIG.float_type.(q_pi[idx]))
        ii = argmax(rand(mnd))
        ii = idx[ii]
    end
    printfmtln("\ntime policy selection= {}\n", Dates.time() - t00)

    policy = IterTools.nth(policies, ii)
    action_ids = collect(zip(policy...))[1]
    action = (; zip(action_names, action_ids)...)
    
    agent.current.action = action

    if true
        printfmtln("\nAction at time {}: {}, ipolicy={}, q_pi= {}, G={}, policy= {}", 
            step_i, action, ii, q_pi[ii], G[ii], policy
        )

        printfmtln("\ncounts of q_pi values:")
        for (k,v) in StatsBase.countmap(round.(q_pi, digits=4))
            if ismissing(k)
                printfmtln("    {}:  {}", "missing", v)
            else
                printfmtln("    {:.4E}:  {}", k, v)
            end
        end
        
        
        if true 
            printfmtln("\nutility= {}, sum= {}", agent.utility[ii, :], round(sum(skipmissing(agent.utility[ii, :])), digits=4))

            printfmtln("\ninfo_gain= {}, sum= {}\n", agent.info_gain[ii, :], round(sum(skipmissing(agent.info_gain[ii, :])), digits=4))

            if !isnothing(agent.info_gain_B)
                printfmtln("\ninfo_gain_B= {}, sum= {}", agent.info_gain_B[ii, :], round(sum(skipmissing(agent.info_gain_B[ii, :])), digits=4))
            end
        end
    end

    #if step_i == 1
    #    @infiltrate; @assert false
    #end
    
    # Push action to agent's history
    #push!(agent.history.actions, action)
    return action
end

