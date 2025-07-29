using ActiveInference
using ActiveInference.Environments
using HDF5
using LinearAlgebra

file_path_gm = "ActiveInference.jl/test/pymdp_cross_val/generative_model_creation/gm_data/gm_matrices.h5"

#############################################
### Loading Generative Model from h5 file ###
#############################################

# A-matrix
A_cross = array_of_any(4)
for i in 1:4
    A_cross[i] = h5read(file_path_gm, "A_cross_$i")
end

# pA-matrix
pA_cross = array_of_any(4)
for i in 1:4
    pA_cross[i] = h5read(file_path_gm, "pA_cross_$i")
end

# B-matrix
B_cross = array_of_any(3)
for i in 1:3
    B_cross[i] = h5read(file_path_gm, "B_cross_$i")
end

# pB-matrix
pB_cross = array_of_any(3)
for i in 1:3
    pB_cross[i] = h5read(file_path_gm, "pB_cross_$i")
end

# C-matrix
C_cross = array_of_any(4)
for i in 1:4
    C_cross[i] = h5read(file_path_gm, "C_cross_$i")
end

# D-matrix
D_cross = array_of_any(3)
for i in 1:3
    D_cross[i] = h5read(file_path_gm, "D_cross_$i")
end

# pD-matrix
pD_cross = array_of_any(3)
for i in 1:3
    pD_cross[i] = h5read(file_path_gm, "pD_cross_$i")
end

################################
### Creating cross val agent ###
################################

settings = Dict("use_param_info_gain" => true,
                "use_states_info_gain" => true,
                "action_selection" => "deterministic",
                "policy_len" => 4)

parameters=Dict{String, Real}("lr_pB" => 0.5,
                              "lr_pA" => 0.5,
                              "lr_pD" => 0.5)

cross_agent = init_aif(A_cross, B_cross, C = C_cross, D = D_cross, pA = pA_cross, pB = pB_cross, pD = pD_cross, settings = settings, parameters = parameters);

#############################################
### Creating and initialising environment ###
#############################################

grid_locations = collect(Iterators.product(1:5, 1:7))
start_loc = (1,1)
cue1_location = (3, 1)
cue2_loc = "L4"
reward_cond = ("BOTTOM")
obs = [1, 1, 1, 1]
location_to_index = Dict(loc => idx for (idx, loc) in enumerate(grid_locations))
actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

cue2_loc_names = ["L1","L2","L3","L4"]
cue2_locations = [(1, 3), (2, 4), (4, 4), (5, 3)]

reward_conditions = ["TOP", "BOTTOM"]
reward_locations = [(2,6), (4,6)]

cue1_names = ["Null";cue2_loc_names]
cue2_names = ["Null", "reward_on_top", "reward_on_bottom"]
reward_names = ["Null", "Cheese", "Shock"]

# Initializing environment
env = EpistChainEnv(start_loc, cue1_location, cue2_loc, reward_cond, grid_locations)

# Getting initial obs
obs = h5read(file_path_gm, "obs")

##########################
### Running simulation ###
##########################

# Time step set to 50 trials
T = 50

# Run simulation
for t in 1:T

    qs = infer_states!(cross_agent, obs)

    update_A!(cross_agent, obs)

    if t != 1
        qs_prev = get_history(cross_agent)["posterior_states"][end-1]
        update_B!(cross_agent, qs_prev)
    end

    if t == 1
        qs_t1 = cross_agent.qs
        update_D!(cross_agent, qs_t1)
    end

    q_pi, G = infer_policies!(cross_agent)

    chosen_action_id = sample_action!(cross_agent)

    movement_id = Int(chosen_action_id[1])
    choice_action = actions[movement_id]

    loc_obs, cue1_obs, cue2_obs, reward_obs = step!(env, choice_action)
    obs = [location_to_index[loc_obs], findfirst(isequal(cue1_obs), cue1_names), findfirst(isequal(cue2_obs), cue2_names), findfirst(isequal(reward_obs), reward_names)]

end


###########################
### Storing the results ###
###########################

# Saving the agent parameters after run for cross validate with pymdp
file_path_results = "ActiveInference.jl/test/pymdp_cross_val/cross_val_results/complete_run_data.h5"

# Storing the A-matrix
h5write(file_path_results, "julia_A_cross_1", cross_agent.A[1])
h5write(file_path_results, "julia_A_cross_2", cross_agent.A[2])
h5write(file_path_results, "julia_A_cross_3", cross_agent.A[3])
h5write(file_path_results, "julia_A_cross_4", cross_agent.A[4])

# Storing the B-matrix
h5write(file_path_results, "julia_B_cross_1", cross_agent.B[1])
h5write(file_path_results, "julia_B_cross_2", cross_agent.B[2])
h5write(file_path_results, "julia_B_cross_3", cross_agent.B[3])

# Storing the D-matrix
h5write(file_path_results, "julia_D_cross_1", cross_agent.D[1])
h5write(file_path_results, "julia_D_cross_2", cross_agent.D[2])
h5write(file_path_results, "julia_D_cross_3", cross_agent.D[3])

# Storing the posterior states
h5write(file_path_results, "julia_qs_1", Float64.(cross_agent.qs[1]))
h5write(file_path_results, "julia_qs_2", Float64.(cross_agent.qs[2]))
h5write(file_path_results, "julia_qs_3", Float64.(cross_agent.qs[3]))
