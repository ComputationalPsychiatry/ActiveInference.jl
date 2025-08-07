# https://pymdp-rtd.readthedocs.io/en/latest/notebooks/clue_chaining_demo.html
# Active Inference Demo: Epistemic Chaining

# include("./grid.jl")
#show(stdout, "text/plain", x)
# @infiltrate; @assert false


#import LogExpFunctions as LEF
import IterTools
import Distributions
import Dates
import StatsBase
import LogExpFunctions as LEF


#using Format
#using Infiltrator
#using Revise
#using Statistics

####################################################################################################


function simulate(model, agent, env, CONFIG, to_label, sim_i)
    
    verbose = true  # for testing

    t0 = Dates.time()

    printfmtln("\n=============\nSimulation= {}, Experiment {}\n=============", sim_i, CONFIG[:experiment])
    
    # use named tuple of named tuples for actions

    obs = step_env!(env, nothing)  # e.g., (loc_obs=7, wall_obs=1, safe_obs=1)
    #observation = [18, 1, 2]

    cell_obs = model.states.loc.labels[obs.loc_obs]
    history_of_locs = [obs.loc_obs]
    history_of_cells = [cell_obs]
    history_of_EFE = []
    history_of_actions = []
    
    action_names = [x.name for x in model.actions]
    

    for step_i in 1:CONFIG.number_simulation_steps
        AI.infer_states!(agent, obs) 
        
        if verbose
            printfmtln("\n-----------\nt={}", step_i)
            for (k,v) in zip(keys(agent.qs), agent.qs)
                printfmtln("    {}: argmax= {}", k, argmax(v))
            end
        else
            if step_i % 100 == 0
                printfmtln("step_i= {}", step_i)
            end
        end

        #AI.update_parameters!(agent)  # no parameter learning in this example
        
        AI.infer_policies!(agent, obs)
        
        #=
        Note that if an explicit graph is used and EFE is over policies, G and q_pi over actions are
        also available and can be queried. 
        =#
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


        if verbose && agent.settings.verbose
            printfmtln("\ncounts of q_pi values = \n{}", StatsBase.countmap(round.(q_pi, digits=6)))
        end

        if verbose && agent.settings.verbose
            idxs = findall(x -> !ismissing(x) && isapprox(x, maximum(skipmissing(q_pi))), q_pi)
            if idxs.size[1] <= 10
                idxs = findall(x -> !ismissing(x) && isapprox(x, maximum(skipmissing(q_pi))), q_pi)
                printfmtln("\nmax q_pi at indexes= {}", idxs)
                printfmtln("\npolicies at max q_pi= \n{}",  
                    [IterTools.nth(policies, ii)[1] for ii in idxs]
                )
            else
                idxs = findall(x -> !ismissing(x) && isapprox(x, maximum(skipmissing(q_pi))), q_pi)
                printfmtln("\nmax q_pi at {} indexes", idxs.size[1])
                printfmtln("\nexample policies at max q_pi= \n{}",  
                    [IterTools.nth(policies, ii)[1] for ii in idxs[1:min(10, idxs.size[1])]]
                )
            end
        end

        # save results in database?
        if false
            save_utility(agent, model, q_pi, step_i, db)
            save_info_gain(agent, model, q_pi, step_i, db)
            save_risk(agent, model, q_pi, step_i, db)
            save_ambiguity(agent, model, q_pi, step_i, db)
            #todo: save info_gain_B
        end

        # sample action
        idx = findall(x -> !ismissing(x), G) 
        if agent.settings.action_selection == :deterministic
            ii = idx[argmax(q_pi[idx])]  # argmax over valid policies
        elseif agent.settings.action_selection == :stochastic
            mnd = Distributions.Multinomial(1, CONFIG.float_type.(q_pi[idx]))
            ii = argmax(rand(mnd))
            ii = idx[ii]
        end

        policy = IterTools.nth(policies, ii)
        action_ids = collect(zip(policy...))[1]
        action = (; zip(action_names, action_ids)...)
        

        agent.current.action = action

        # Push action to agent's history
        #push!(agent.history.actions, action)

        if verbose
            printfmtln("\nAction at time {}: {}, ipolicy={}, q_pi= {}, G={}", 
                step_i, action, ii, q_pi[ii], G[ii]
            )
            
            if isnothing(agent.G_actions) && agent.settings.verbose
                printfmtln("\nutility= {}, sum= {}", agent.utility[ii, :], round(sum(skipmissing(agent.utility[ii, :])), digits=4))

                printfmtln("\ninfo_gain= {}, sum= {}", agent.info_gain[ii, :], round(sum(skipmissing(agent.info_gain[ii, :])), digits=4))

                if !isnothing(agent.info_gain_B)
                    printfmtln("\ninfo_gain_B= {}, sum= {}", agent.info_gain_B[ii, :], round(sum(skipmissing(agent.info_gain_B[ii, :])), digits=4))
                end
            end
        end

        # check G values
        #@infiltrate; @assert false

        # record EFE choice
        if false && isnothing(agent.G_actions)
            push!(history_of_EFE, [
                sum(agent.info_gain[ii, :]), 
                sum(agent.utility[ii, :]), 
                sum(agent.risk[ii, :]), 
                sum(agent.ambiguity[ii, :]), 
                sum(agent.info_gain[ii, :]) + sum(agent.utility[ii, :]), 
            ])
        end
        push!(history_of_actions, action)
        

        obs = step_env!(env, action)
        cell_obs = model.states.loc.labels[obs.loc_obs]
        #push!(history_of_observations, obs.loc_obs)
        push!(history_of_cells, cell_obs)

        if verbose
            printfmtln("Grid location at time {}, after action: {}", step_i, obs)
        end
        
        printfmtln("\nsimulation time= {}\n", round((Dates.time() - t0) , digits=2))
        
    end

    printfmtln("\nsimulation time= {}\n", round((Dates.time() - t0) , digits=2))
    
    # make animation
    #anim = Animation()
    #for p in plots
    #    Plots.frame(anim, p)
    #end
    #Plots.gif(anim, gif_name, fps=1)
    
    results = Dict(
        :loc_id => history_of_locs,
        :EFE => history_of_EFE,
        :actions => history_of_actions,
        #:sq_error => history_of_sq_error,
        #:r => history_of_r,
        :cells => history_of_cells
    )

    #@infiltrate; @assert false

    return results

end