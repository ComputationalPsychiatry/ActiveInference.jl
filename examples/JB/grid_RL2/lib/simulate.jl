# https://pymdp-rtd.readthedocs.io/en/latest/notebooks/clue_chaining_demo.html
# Active Inference Demo: Epistemic Chaining

# include("./grid.jl")
#show(stdout, "text/plain", x)
# @infiltrate; @assert false


#import LogExpFunctions as LEF
import IterTools
import Plots.Animation as Animation
import Distributions
import Dates
#import SQLite
import StatsBase
import LogExpFunctions as LEF


include("./make_env.jl")


#using Format
#using Infiltrator
#using Revise
#using Statistics

####################################################################################################


function simulate(model, agent, env, CONFIG, to_label, sim_i)
    
    verbose = agent.settings.verbose
    verbose = true  # for testing

    t0 = Dates.time()

    printfmtln("\n=============\nSimulation= {}, Experiment {}\n=============", sim_i, CONFIG[:experiment])
    
    
    #db_name = format("./dbs/{}_sim{}.sqlite", CONFIG[:experiment], sim_i)
    gif_name = format("./gifs/{}_sim{}.gif", CONFIG[:experiment], sim_i)
    plot_title = format("{}, Sim={}", CONFIG[:experiment], sim_i)
    
    #if isfile(db_name)
    #    rm(db_name)  # remove if db exists
    #end
    #db = SQLite.DB(db_name)

    plots = Vector{Plots.Plot}()
    push!(plots, plot_grid(CONFIG, to_label, plot_title, sim_i, 0, CONFIG[:walls]))
    
    current_loc = reset_env!(env)
    
    # use named tuple of named tuples for actions

    if length(model.actions[1].labels) == 1
        obs = step_env!(env, (move=(5),))
    else
        # two actions
        #@infiltrate; @assert false
        obs = step_env!(env, (move_vert=3, move_horz=3))  # e.g., (loc_obs=7,)
    end

    cell_obs = model.states.loc.labels[obs.loc_obs]
    history_of_locs = [obs.loc_obs]
    history_of_cells = [cell_obs]
    history_of_EFE = []
    history_of_actions = []
    history_of_sq_error = []
    history_of_r = []

    action_names = [x.name for x in model.actions]
    
    for step_i in 1:CONFIG.number_simulation_steps
        AI.infer_states!(agent, obs) 
        
        if verbose
            printfmtln("\n-----------\nt={}", step_i)
            for (k,v) in zip(keys(agent.qs_current), agent.qs_current)
                printfmtln("    {}: argmax= {}", k, argmax(v))
            end
        else
            if step_i % 100 == 0
                printfmtln("step_i= {}", step_i)
            end
        end

        AI.update_parameters!(agent)
        #@infiltrate; @assert false

        AI.infer_policies!(agent, obs)
        
        #=
        If policy inference is over actions, then use agent.q_pi and agent.G_actions, both over 
        actions. If inference is over policies, then use agent.q_pi and agent.G, both over policies.
        =#

        if !isnothing(agent.G_actions)
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

        if verbose    
            printfmtln("\ncounts of q_pi values = \n{}", StatsBase.countmap(round.(q_pi, digits=6)))
        end

        idxs = findall(x -> !ismissing(x) && isapprox(x, maximum(skipmissing(q_pi))), q_pi)
        if verbose && idxs.size[1] <= 10
            printfmtln("\nmax q_pi at indexes= {}", idxs)
            printfmtln("\npolicies at max q_pi= \n{}",  
                [IterTools.nth(policies, ii)[1] for ii in idxs]
            )
        elseif verbose
            printfmtln("\nmax q_pi at {} indexes", idxs.size[1])
            printfmtln("\nexample policies at max q_pi= \n{}",  
                [IterTools.nth(policies, ii)[1] for ii in idxs[1:min(10, idxs.size[1])]]
            )
        end

        printfmtln("\nsimulation time= {}\n", round((Dates.time() - t0) , digits=2))
        #@infiltrate; @assert false
        
        # save results in database?
        if false
            save_utility(agent, model, q_pi, step_i, db)
            save_info_gain(agent, model, q_pi, step_i, db)
            save_risk(agent, model, q_pi, step_i, db)
            save_ambiguity(agent, model, q_pi, step_i, db)
            #todo: save info_gain_B
        end

        
        # selection of policy ii
        # custom sample action
        if agent.settings.action_selection == :deterministic
            ii = argmax(q_pi)  # any path
        elseif agent.settings.action_selection == :stochastic
            idx = findall(x -> !ismissing(x), G) 
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

        if verbose
            printfmtln("\nAction at time {}: {}, ipolicy={}, q_pi= {}, G={}", 
                step_i, action, ii, q_pi[ii], G[ii]
            )
            
            if isnothing(agent.G_actions)
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
        if isnothing(agent.G_actions)
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

        sq_error = Statistics.sum((CONFIG.B_true .- agent.model.states.loc.B) .^2)
        push!(history_of_sq_error, sq_error)
        
        if sq_error == 0
            break
        end
            
        if false
            push!(plots, plot_grid(CONFIG, to_label, plot_title, sim_i, step_i, CONFIG[:walls], locations=history_of_cells))
        else
            if step_i % max(Int(round(CONFIG.number_simulation_steps/10)), 1) == 0
                plot_visited(CONFIG, to_label, plot_title, sim_i, step_i, CONFIG[:walls], history_of_cells)
                plot_sq_error(CONFIG, sim_i, step_i, history_of_sq_error)
                #@infiltrate; @assert false
            end
        end
        
        #print(f'Reward at time {t}: {reward_obs}')

        #if t == 2
        #    @infiltrate; @assert false
        #end   
        #@infiltrate; @assert false
    end

    printfmtln("\nsimulation time= {}\n", round((Dates.time() - t0) , digits=2))
    @infiltrate; @assert false

    # make animation
    anim = Animation()
    for p in plots
        Plots.frame(anim, p)
    end
    Plots.gif(anim, gif_name, fps=1)
    
    results = Dict(
        :loc_id => history_of_locs,
        :EFE => history_of_EFE,
        :actions => history_of_actions,
        :sq_error => history_of_sq_error,
        :cells => history_of_cells
    )

    # 1000 steps in 4419 sec = 73.6 minutes
    @infiltrate; @assert false

    return results

end