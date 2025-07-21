# using new package: dev /home/john/Dev/ActInf.jl/ActiveInference.jl
# https://pymdp-rtd.readthedocs.io/en/latest/notebooks/clue_chaining_demo.html
# Active Inference Demo: Epistemic Chaining

# include("./gridA.jl")
#show(stdout, "text/plain", x)
# @infiltrate; @assert false


module Assess


import Plots
import Random
import Distributions
import DataFrames as DFS
import CSV
import Serialization
import IterTools

using PyCall
using FreqTables
using Statistics
import LinearAlgebra as LA
import StatsBase
import LogExpFunctions as LEF

using Format
using Infiltrator
using Revise

include("./lib/metrics.jl")
include("./lib/make_plots.jl")

DF = DFS.DataFrame
#Random.seed!(51233) # Set random seed for reproducibility
Plots.scalefontsizes()
Plots.scalefontsizes(0.8)

@pyimport pickle

# workon thermo_orig (for calling PyInform)
# Pkg.add("PyCall")
# julia> Pkg.build("PyCall")

####################################################################################################

py"""
import pickle
 
def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data
"""


# -------------------------------------
function assess_results(CONFIG, agent, model, to_label)
    
    AIs = Serialization.deserialize(format("{}_history.ser", CONFIG[:experiment]))
    B_true = Serialization.deserialize(format("{}_B.ser", CONFIG[:experiment]))

    load_pickle = py"load_pickle"

    fns1 = [
        "/home/john/Dev/Mchristos2/empowerment_extended/results_emp1.pkl",
        "/home/john/Dev/Mchristos2/empowerment_extended/results_emp2.pkl",
        "/home/john/Dev/Mchristos2/empowerment_extended/results_emp3.pkl",
    ]

    fns2 = [
        "/home/john/Dev/Mchristos2/empowerment_extended/results_no_emp1.pkl",
        "/home/john/Dev/Mchristos2/empowerment_extended/results_no_emp2.pkl",
        "/home/john/Dev/Mchristos2/empowerment_extended/results_no_emp3.pkl",
    ]
    
    RL1 = []
    for fn in fns1
        data = load_pickle(fn)
        push!(RL1, data)
    end

    RL2 = []
    for fn in fns2
        data = load_pickle(fn)
        push!(RL2, data)
    end

    """
    infil> keys(RL1[1])
        "B"
        "visited"
        "final_t"
        "history_cells"
        "history_sq_error"
        "D_emp"
        "tau"
        "history_emp"
        "D_mod"
    
    infil> keys(AIs[1])
        :EFE
        :actions
        :obs
        :sq_error
        :loc_id
    """
    
    B = B_true[1]  # (next_state, curr_state, action_id)
    E = zeros(model.grid_dims)
    
    for y in 1:model.grid_dims[1]
        for x in 1:model.grid_dims[2]
            s = findfirst(z -> z == [y,x], model.cells)
            nstep_actions = collect(Iterators.product(repeat([1:B.size[end]], model.policy_length)...))[:]
            Bn = zeros(B.size[1], B.size[2], length(nstep_actions))
            for (i, an) in enumerate(nstep_actions)
                Bn[:,:,i] = mapreduce(a -> B[:,:,a], *, an)
                @assert IterTools.nth(model.policies.policy_iterator, i)[1] == an
            end
            emp = Metrics.blahut_arimoto(permutedims(Bn[:,s,:]) .+ 1e-9)[1]
            E[y,x] = emp
        end
    end
    
    plot_empowerment(CONFIG, to_label, CONFIG[:walls], E)
    
    plt = Plots.plot(title="Sum of Squared Error, Learning B Matrix")
    
    error_ai_mean = mean(hcat([AIs[ii][:sq_error] for ii in 1:3]))
    error_rl1_mean = mean(hcat([RL1[ii]["history_sq_error"] for ii in 1:3]))
    error_rl2_mean = mean(hcat([RL2[ii]["history_sq_error"] for ii in 1:3]))
    

    Plots.plot!(error_ai_mean, label="ActInf Agent")
    Plots.plot!(error_rl1_mean, label="Emp RL Agent")
    Plots.plot!(error_rl2_mean, label="Non-Emp RL Agent")
    
    Plots.savefig("./pngs/RL1_sq_error.png")
    
    # for RL agent, position 13 -> cell [1,3)
    # for RL agent, position 32 -> cell [3,2)]

    idx1 = permutedims(reshape(collect(1:100), 10,10))
    idx2 = permutedims(reshape(collect(100:-1:1), 10,10))[:, 10:-1:1]
    conv = permutedims(idx1)[10:-1:1, :]
    
    ai_emp_sum = sum([E[model.cells[ii]...] for ii in AIs[1][:loc_id]])
    ai_emp_mean = mean([E[model.cells[ii]...] for ii in AIs[1][:loc_id]])
    
    printfmtln("\nActInf empowerment sum= {}, mean= {}, n=10,000, sim=1",
        round(ai_emp_sum, digits=0),
        round(ai_emp_mean, digits=3), 
    )

    
    ai_emp_sum_5300 = sum([E[model.cells[ii]...] for ii in AIs[1][:loc_id]][1:5300])
    ai_emp_mean_5300 = mean([E[model.cells[ii]...] for ii in AIs[1][:loc_id]][1:5300])
    printfmtln("\nActInf empowerment sum= {}, mean= {}, n=1:5300, sim=1",
        round(ai_emp_sum_5300, digits=0),
        round(ai_emp_mean_5300, digits=3), 
    )
    
    
    means = []
    sums = []
    ns = []
    for ii in 1:3
        emp = RL1[ii]["history_emp"]
        n = findfirst(x -> isapprox(x, 0), emp)
        push!(ns, n)
        emp = emp[1:n]
        emp_sum = sum(emp)
        emp_mean = mean(emp)
        push!(means, emp_mean)
        push!(sums, emp_sum)
    end

    printfmtln("\nEmpowered RL agent, empowerment sum= {}, mean= {}, ns={}, average of 3 simulations",
        round(mean(sums), digits=0),
        round(mean(means), digits=3),
        round(mean(ns), digits=0), 
    )
    
    means2 = []
    sums2 = []
    ns2 = []
    for ii in 1:3
        emp = RL2[ii]["history_emp"]
        n = findfirst(x -> isapprox(x, 0), emp)
        push!(ns2, n)
        emp = emp[1:n]
        emp_sum = sum(emp)
        emp_mean = mean(emp)
        push!(means2, emp_mean)
        push!(sums2, emp_sum)
    end

    printfmtln("\nNon-Empowered RL agent, empowerment sum= {}, mean= {}, ns={}, average of 3 simulations",
        round(mean(sums2), digits=0),
        round(mean(means2), digits=3),
        round(mean(ns2), digits=0), 
    )

    """
    ActInf empowerment sum= 38824.0, mean= 3.882, n=10,000, sim=1
    ActInf empowerment sum= 20532.0, mean= 3.874, n=1:5300, sim=1
    Empowered RL agent, empowerment sum= 20776.0, mean= 3.941, ns=5271.0, average of 3 simulations
    Non-Empowered RL agent, empowerment sum= 21809.0, mean= 3.802, ns=5735.0, average of 3 simulations
    """

    loc_cnts = StatsBase.countmap(AIs[1][:loc_id])
    
    printfmtln("\nActInf, cnts of visits, mean= {}, std= {}", 
        round(mean(values(loc_cnts )), digits=3),
        round(std(values(loc_cnts )), digits=3),
    )
    
    loc_cnts = StatsBase.countmap(AIs[1][:loc_id][1:5300])
    printfmtln("\nActInf, cnts of visits, 1:5300, mean= {}, std= {}", 
        round(mean(values(loc_cnts)), digits=3),
        round(std(values(loc_cnts)), digits=3),
    )

    cnt_mean = []
    cnt_std = []
    for ii in 1:3
        cnts = RL1[ii]["visited"][:]
        push!(cnt_mean, mean(cnts))
        push!(cnt_std, std(cnts))
    end
    printfmtln("\nRL1, cnts of visits, mean= {}, std= {}", 
        round(mean(cnt_mean), digits=3),
        round(mean(cnt_std), digits=3),
    )

    cnt_mean2 = []
    cnt_std2 = []
    for ii in 1:3
        cnts = RL2[ii]["visited"][:]
        push!(cnt_mean2, mean(cnts))
        push!(cnt_std2, std(cnts))
    end
    printfmtln("\nRL2, cnts of visits, mean= {}, std= {}", 
        round(mean(cnt_mean2), digits=3),
        round(mean(cnt_std2), digits=3),
    )

    """
    ActInf, cnts of visits, mean= 100.01, std= 1.267
    ActInf, cnts of visits, 1:5300, mean= 53.0, std= 2.243
    RL1, cnts of visits, mean= 52.7, std= 15.82
    RL2, cnts of visits, mean= 57.343, std= 34.054
    """



    @infiltrate; @assert false
    for (loc, cnt) in locations
        data[loc...] = cnt
    end


    @infiltrate; @assert false

    B1 = B_true[1]  # (100, 100,5)
    B21 = RL1[1]["B"]  # (100, 5, 100)
    B22 = RL2[1]["B"]  # (100, 5, 100)
    @assert all(B21 .== B22)
    
    B21 = permutedims(B21, (1,3,2))
    B22 = permutedims(B22, (1,3,2))

    for i in 1:5
        @infiltrate; @assert false
        B21[:,:,i] = B21[:,:,i][conv]
        B22[:,:,i] = B22[:,:,i][conv...]
    end

    # B1[4,14,1], B21r[4,14,1] = (0,1); cell [2,4] to [1,4], moving up

    
    @infiltrate; @assert false
    
    
end

end  # -- module

#Assess.run()