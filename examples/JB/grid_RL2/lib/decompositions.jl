
#show(stdout, "text/plain", x)
# @infiltrate; @assert false


function save_utility(agent, model, q_pi, t, db)

    printfmtln("\nUtility: t= {}, size G={}, G>0= {}", t, agent.G.size[1], sum(agent.G .!= 0)) 

    gg = agent.utility .+ agent.info_gain
    if false
        #for ii in 1: gg.size[1]
        for ii in []

            if all(ismissing.(gg[ii,:]))
                continue
            end
            
            policy = IterTools.nth(agent.policies.policy_iterator, ii)[1]
            policy = [model.action_deps[:move][x] for x in policy]
            printfmtln("\n    {}, util={}, info={}, \n    both={}, sum= {}, policy= {}", 
                ii,
                round.(agent.utility[ii,:], digits=4),
                round.(agent.info_gain[ii,:], digits=4), 
                round.(gg[ii,:], digits=4), 
                round(sum(skipmissing(gg[ii,:])), digits=4),
                policy
            )
        end 
    end

    if true
        # display utility info
        unique_utility = sortslices(unique(agent.utility, dims=1), dims=1, by= x-> sum(x))
        
        if unique_utility.size[1] < 15
            printfmtln("\nunique utility, min={}, max= {}:", 
                minimum(skipmissing(sum(unique_utility, dims=2))), 
                maximum(skipmissing(sum(unique_utility, dims=2)))
            )
            display(unique_utility)
        else
            printfmtln("\nutility: size uniq= {}, min sum={}, max sum= {}", 
                unique_utility.size[1], 
                minimum(skipmissing(sum(unique_utility, dims=2))), 
                maximum(skipmissing(sum(unique_utility, dims=2)))
            )
        end

        printfmtln("\nunique utility values= {}", round.(unique(vcat(agent.utility...)), digits=4 ))

        # display examples of each
        println("\nexamples of unique utility:")
        for ii in 1:unique_utility.size[1]
            if all(ismissing.(unique_utility[ii,:]))
                continue
            end
            idx = nothing
            for jj in 1:agent.utility.size[1]
                if all(ismissing.(agent.utility[jj,:]))
                    continue
                end
                
                good1 = .!ismissing.(unique_utility[ii,:])
                good2 = .!ismissing.(agent.utility[jj,:])
                if all(good1 .== good2)
                    if all(unique_utility[ii,good1] .== agent.utility[jj,good2])
                        idx = jj
                        break
                    end
                end
            end
            policy = IterTools.nth(agent.policies.policy_iterator, idx)[1]
            policy = [model.action_deps[:move][x] for x in policy]
            printfmtln("    ii={}, utility={}, sum= {}, policy= {}",
                idx, round.(unique_utility[ii,:], digits=4), 
                round.(sum(skipmissing(unique_utility[ii,:])), digits=4), 
                policy
            )
            #@infiltrate; @assert false
        end

        if unique_utility.size[1] > 1
            #@infiltrate; @assert false
            idx = findall(x -> !ismissing(x), sum(agent.utility, dims=2)[:,1])
            util = agent.utility[idx,:]
            ids = collect(1:agent.utility.size[1])[idx]

            mean_ = mean_ = [mean(skipmissing(x)) for x in eachrow(util)]
            idx2 = reverse(sortperm(mean_))
            minn = minimum(mean_)
            println("\ntop utility and policy:")
            for (ii, jj) in enumerate(idx2[1: minimum([20, length(idx2)])])
                #if mean(skipmissing(util[jj,:])) == minn
                #    break
                #end
                policy = IterTools.nth(agent.policies.policy_iterator, idx[jj])[1]
                policy = [model.action_deps[:move][x] for x in policy]
                printfmtln("    ii={}, utility={}, policy_id={}, policy={}, sum={}",
                    ii, round.(util[jj,:], digits=4), idx[jj], policy, round(sum(skipmissing(util[jj,:])), digits=4)
                )
            end
        end
    end
    
    n,m = agent.utility.size
    utility = round.(agent.utility, digits=4)
    df = DFS.DataFrame(utility, [format("step_{}", x) for x in 1:m]) 
    df[!, "Sum"] .= [round.(sum(skipmissing(utility[ii,:])), digits=4) for ii in 1:n]

    # todo: allow for multiple actions per policy
    df[!, "Policy"] = [string([model.action_deps[:move][x] for x in policy[1]]) for policy in agent.policies.policy_iterator]
    df[!, "Util+Info"] = [round.(sum(skipmissing(gg[ii,:])), digits=4) for ii in 1:n]
    df[!, "G"] = round.(agent.G, digits=4)
    DFS.insertcols!(df, 1, :ID => 1:n)
    DFS.insertcols!(df, 2, :t => t)
    DFS.sort!(df, ["Sum", :ID])
    
    SQLite.load!(df, db, "utility")
    #@infiltrate; @assert false
end



# --------------------------------------------------------------------------------------------------
function save_info_gain(agent, model, q_pi, t, db)
    if true
        gg = agent.utility .+ agent.info_gain

        unique_info_gain = sortslices(unique(agent.info_gain, dims=1), dims=1, by= x-> sum(x))

        if unique_info_gain.size[1] < 15
            printfmtln("\nunique info_gain:")
            display(unique_info_gain)
        else
            printfmtln("\ninfo_gain: size uniq= {}, min={}, max= {}", 
                unique_info_gain.size[1], minimum(skipmissing(sum(unique_info_gain, dims=2))), maximum(skipmissing(sum(unique_info_gain, dims=2)))
            )
        end

        printfmtln("\nunique info_gain values= {}", round.(unique(vcat(agent.info_gain...)), digits=4 ))

        if unique_info_gain.size[1] > 1
            idx = findall(x -> !isapprox(x, 0.0), agent.G)
            gain = agent.info_gain[idx,:]
            ids = collect(1:agent.info_gain.size[1])[idx]

            mean_ = mean_ = [mean(skipmissing(x)) for x in eachrow(gain)]
            idx2 = reverse(sortperm(mean_))
            minn = minimum(mean_)
            
            println("\ntop info_gain and policy:")
            for (ii, jj) in enumerate(idx2[1:minimum([20, length(idx2)])])
                #if mean(skipmissing(gain[jj,:])) == minn
                #    break
                #end
                policy = IterTools.nth(agent.policies.policy_iterator, idx[jj])[1]
                policy = [model.action_deps[:move][x] for x in policy]
                #@infiltrate; @assert false
                printfmtln("    ii={}, info_gain={}, policy_id= {}, policy= {},  sum={}",
                    ii, round.(gain[jj,:], digits=4), idx[jj], policy, round(sum(skipmissing(gain[jj,:])), digits=4)
                )
            end
            #@infiltrate; @assert false
        end
    end    

    n,m = agent.info_gain.size
    info_gain = round.(agent.info_gain, digits=4)
    df = DFS.DataFrame(info_gain, [format("step_{}", x) for x in 1:m]) 
    df[!, "Sum"] .= [round.(sum(skipmissing(info_gain[ii,:])), digits=4) for ii in 1:n]

    # todo: allow for multiple actions per policy
    df[!, "Policy"] = [string([model.action_deps[:move][x] for x in policy[1]]) for policy in agent.policies.policy_iterator]
    df[!, "Util+Info"] = [round.(sum(skipmissing(gg[ii,:])), digits=4) for ii in 1:n]
    df[!, "G"] = round.(agent.G, digits=4)
    DFS.insertcols!(df, 1, :ID => 1:n)
    DFS.insertcols!(df, 2, :t => t)
    DFS.sort!(df, ["Sum", :ID])
    
    SQLite.load!(df, db, "info_gain")
    #@infiltrate; @assert false
end


# --------------------------------------------------------------------------------------------------
function save_risk(agent, model, q_pi, t, db)
    gg = agent.risk .+ agent.ambiguity

    n,m = agent.risk.size
    risk = round.(agent.risk, digits=4)
    df = DFS.DataFrame(risk, [format("step_{}", x) for x in 1:m]) 
    df[!, "Sum"] .= [round.(sum(skipmissing(risk[ii,:])), digits=4) for ii in 1:n]

    # todo: allow for multiple actions per policy
    df[!, "Policy"] = [string([model.action_deps[:move][x] for x in policy[1]]) for policy in agent.policies.policy_iterator]
    df[!, "Risk+Ambiguity"] = [round.(sum(skipmissing(gg[ii,:])), digits=4) for ii in 1:n]
    df[!, "G"] = round.(agent.G, digits=4)
    DFS.insertcols!(df, 1, :ID => 1:n)
    DFS.insertcols!(df, 2, :t => t)
    DFS.sort!(df, ["Sum", :ID])
    
    SQLite.load!(df, db, "risk")
    #@infiltrate; @assert false
end


# --------------------------------------------------------------------------------------------------
function save_ambiguity(agent, model, q_pi, t, db)
    gg = agent.risk .+ agent.ambiguity

    n,m = agent.ambiguity.size
    ambiguity = round.(agent.ambiguity, digits=4)
    df = DFS.DataFrame(ambiguity, [format("step_{}", x) for x in 1:m]) 
    df[!, "Sum"] .= [round.(sum(skipmissing(ambiguity[ii,:])), digits=4) for ii in 1:n]

    # todo: allow for multiple actions per policy
    df[!, "Policy"] = [string([model.action_deps[:move][x] for x in policy[1]]) for policy in agent.policies.policy_iterator]
    df[!, "Risk+Ambiguity"] = [round.(sum(skipmissing(gg[ii,:])), digits=4) for ii in 1:n]
    df[!, "G"] = round.(agent.G, digits=4)
    DFS.insertcols!(df, 1, :ID => 1:n)
    DFS.insertcols!(df, 2, :t => t)
    DFS.sort!(df, ["Sum", :ID])
    
    SQLite.load!(df, db, "ambiguity")
    #@infiltrate; @assert false
end
