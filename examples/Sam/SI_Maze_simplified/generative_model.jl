include("environment.jl")
using LogExpFunctions

# possible observations:
# [Location, Open/Closed, Safe/Aversive]
states = [81]
observations = [81,2,2]
controls=[2]

A,B,C,D = create_matrix_templates(states, observations, controls, 1, "zeros")

################## State-Observation Mapping ##################
# Location Modality 
A[1] += I(81)
#heatmap(A[1], color=cgrad(:greys), yflip=true, aspect_ratio=:equal)

#=
# Stochastic A-matrix
A[1] += 0.8 * I(81)
A[1] .+= (1.0 - 0.8) / 81
heatmap(A[1], color=cgrad(:greys), yflip=true, aspect_ratio=:equal)
=#

# "Wall" Modality
MAZE_vec = vec(MAZE)
A[2] = vcat((1 .- MAZE_vec'), MAZE_vec')
#heatmap(A[2], color=cgrad(:greys), yflip=true, yticks=1:2,xticks=1:8:81)

# Preference Modality
# Probabilities of observing Safe Location
A[3][1, :] = vec(PREFERENCES)

# Probabilities of observing Aversive Location
A[3][2, :] = 1 .- vec(PREFERENCES)
#heatmap(A[3], color=cgrad(:greys), yflip=true)
######################## Transitions ########################
# [UP, RIGHT]
include("construct_B.jl")
B[1] = construct_B_matrices(MAZE, 2)

#=
# Stochastic B-matrix
temperature = 0.15  

# Apply softmax to each column of the first transition matrix
for col in 1:size(B[1], 2)
    B[1][:, col, 1] = softmax(B[1][:, col, 1] ./ temperature)
end

# Apply softmax to each column of the second transition matrix
for col in 1:size(B[1], 2)
    B[1][:, col, 2] = softmax(B[1][:, col, 2] ./ temperature)
end

B[1][:,:,1]
B[1][:,:,2]
=#

######################## Preferences ########################
C[1] # No preference for Location
C[2] # No preference for Wall 
C[3] = [1.0, -1.0] # Preference for Safe Locations

################# Prior over initial state ##################
D[1] = onehot(18, 81) # Start at location 18

####################################################################
######################### SIMULATE BEHAVIOR ########################
####################################################################

parameters = Dict("gamma" => 1.0)

struct MetaModel
    obs_deps::Union{Nothing, NamedTuple{syms, <:NTuple{n, Vector{Symbol}}}} where {syms,n}
    obs_dims::Union{Nothing, NamedTuple{syms, <:NTuple{n, Vector{Int64}}}} where {syms,n}
    state_deps::Union{Nothing, NamedTuple{syms, <:NTuple{n, Vector{Symbol}}}} where {syms,n}
    state_dims::Union{Nothing, NamedTuple{syms, <:NTuple{n, Vector{Int64}}}} where {syms,n}
    action_deps::Union{Nothing, NamedTuple{syms, <:NTuple{n, Vector{Symbol}}}} where {syms,n}
    action_dims::Union{Nothing, NamedTuple{syms, <:NTuple{n, Vector{Int64}}}} where {syms,n}
    policies
end

struct Policies
    policy_iterator::Union{Vector{Tuple}, Base.Iterators.ProductIterator}
    action_contexts::Dict{Symbol, Dict{Symbol, Any}}
    number_policies::Int64
end

obs_deps = (loc_obs = [:loc_obs, :loc], wall_obs = [:wall_obs, :loc], pref_obs = [:pref_obs, :loc])
obs_dims = (loc_obs = [81, 81], wall_obs = [2, 81], pref_obs = [2, 81])
state_deps = (loc = [:loc, :loc, :move],)
state_dims = (loc = [81, 81, 2],)
action_deps = (move = [:UP, :RIGHT],)
action_dims = (move = [2],)