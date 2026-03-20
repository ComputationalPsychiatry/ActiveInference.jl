using ActiveInference
using HDF5
using Random

# Setting seed for reproducibility purposes
Random.seed!(246)

##########################################
### Generating random generative model ###
##########################################

# Setting number of states, observations and controls for the generative model
n_states = [35, 4, 2]
n_obs = [35, 5, 3, 3]
n_controls = [5, 1, 1]
policy_length = 1

# Using function for generating A and B matrices with random inputs
A_cross, B_cross = create_matrix_templates(n_states, n_obs, n_controls, policy_length, "random");

# Generating random C matrix
C_cross = array_of_any(4)

C_cross[1] = Float64.(rand(1:10, 35))
C_cross[2] = Float64.(rand(1:10, 5))
C_cross[3] = Float64.(rand(1:10, 3))
C_cross[4] = Float64.(rand(1:10, 3))

# Generating random D matrix
D_cross = array_of_any(3)

D_cross[1] = rand(1:10, 35)
D_cross[2] = rand(1:10, 4)
D_cross[3] = rand(1:10, 2)

D_cross = normalize_arrays(D_cross)

# Setting file path for h5 file containing the dataframes

file_path_gm = "ActiveInference.jl/test/pymdp_cross_val/generative_model_creation/gm_data/gm_matrices.h5"

# Storing the layers for each modality in A matrix in an h5 file. HDF5 can't take an array of arrays 
h5write(file_path_gm, "A_cross_1", A_cross[1])
h5write(file_path_gm, "A_cross_2", A_cross[2])
h5write(file_path_gm, "A_cross_3", A_cross[3])
h5write(file_path_gm, "A_cross_4", A_cross[4])

# Storing the layers for each factor in B matrix in an h5 file. HDF5 can't take an array of arrays 
h5write(file_path_gm, "B_cross_1", B_cross[1])
h5write(file_path_gm, "B_cross_2", B_cross[2])
h5write(file_path_gm, "B_cross_3", B_cross[3])

# Storing the layers for each modality in C matrix in an h5 file. HDF5 can't take an array of arrays
h5write(file_path_gm, "C_cross_1", C_cross[1])
h5write(file_path_gm, "C_cross_2", C_cross[2])
h5write(file_path_gm, "C_cross_3", C_cross[3])
h5write(file_path_gm, "C_cross_4", C_cross[4])

# Storing the layers for each factor in D matrix in an h5 file. HDF5 can't take an array of arrays
h5write(file_path_gm, "D_cross_1", D_cross[1])
h5write(file_path_gm, "D_cross_2", D_cross[2])
h5write(file_path_gm, "D_cross_3", D_cross[3])

#####################################
### Generating random observation ###
#####################################
obs = Int[]
for (i, j) in enumerate(n_obs)
    observation = rand(1:j)
    push!(obs, observation) 
end

h5write(file_path_gm, "obs", obs)

#################################################
### Generating random qs_prev + random action ###
#################################################
qs_prev = array_of_any(3)

for (i, j) in enumerate(n_states)
    qs_prev[i] = rand(1:10, j)
end
qs_prev = normalize_arrays(qs_prev)

h5write(file_path_gm, "qs_prev_1", qs_prev[1])
h5write(file_path_gm, "qs_prev_2", qs_prev[2])
h5write(file_path_gm, "qs_prev_3", qs_prev[3])

action = [rand(1:5), 1, 1]
h5write(file_path_gm, "action", action)

#################################################################
### Generating dirichlet distributions for learning functions ###
#################################################################

# pA
# setting the concentration parameter arbitrarily
pA_cross = deepcopy(A_cross)
for i in 1:length(pA_cross)
    pA_cross[i] = pA_cross[i] .* 10
end

h5write(file_path_gm, "pA_cross_1", pA_cross[1])
h5write(file_path_gm, "pA_cross_2", pA_cross[2])
h5write(file_path_gm, "pA_cross_3", pA_cross[3])
h5write(file_path_gm, "pA_cross_4", pA_cross[4])

# pB
pB_cross = deepcopy(B_cross)
for i in 1:length(pB_cross)
    pB_cross[i] = pB_cross[i] .* 10
end

h5write(file_path_gm, "pB_cross_1", pB_cross[1])
h5write(file_path_gm, "pB_cross_2", pB_cross[2])
h5write(file_path_gm, "pB_cross_3", pB_cross[3])

# pD
pD_cross = deepcopy(D_cross)

for (i, j) in enumerate(n_states)
    pD_cross[i] = pD_cross[i] .* 10
end

h5write(file_path_gm, "pD_cross_1", pD_cross[1])
h5write(file_path_gm, "pD_cross_2", pD_cross[2])
h5write(file_path_gm, "pD_cross_3", pD_cross[3])



