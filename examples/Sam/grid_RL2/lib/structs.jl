#show(stdout, "text/plain", x)
# @infiltrate; @assert false

#import MetaGraphsNext as MGN

using Format
using Infiltrator
using Revise


####################################################################################################

# Data
struct Data
    name::Symbol
    labels::Union{Nothing, Vector}
    loc::Union{Nothing, Vector, Tuple}
    dependencies::Union{Nothing, Vector}
    action_dependencies::Union{Nothing, Vector}
    n
end


function Data(name; labels=nothing, loc=nothing, dependencies=nothing, action_dependencies=[])

    if !isnothing(labels)
        n = length(labels)
    elseif !isnothing(loc)
        n = length(loc)  
    else 
        n = nothing
    end

    return Data(
        name,
        labels,
        loc,
        dependencies,
        action_dependencies,
        n
    )
end


# States
@kwdef struct States
    loc::Data
end


# Observations
@kwdef struct Observations
    loc_obs::Data
end


# Actions
@kwdef struct Actions
    #move::Data
    move_vert::Data
    move_horz::Data

end 


# Preferences
@kwdef struct Preferences
    loc_pref::Data
end


# Model is holder for user data, some of which will be passed to agent or MetaModel
@kwdef mutable struct Model
    grid_dims::Tuple{Int64, Int64}
    cells::Vector{Vector{Int64}}
    states::States
    observations::Observations
    actions::Actions
    preferences::Preferences
    start::Union{Nothing, Data}
    policy_length::Int64
        
    A::Union{Nothing, Vector} = nothing
    B::Union{Nothing, Vector} = nothing
    B_true::Union{Nothing, Vector} = nothing
    C::Union{Nothing, Vector} = nothing
    D::Union{Nothing, Vector} = nothing
    
    A_deps::Union{Nothing, NamedTuple{syms, <:NTuple{n, Vector{Symbol}}}} where {syms,n} = nothing
    A_dims::Union{Nothing, NamedTuple{syms, <:NTuple{n, Vector{Int64}}}} where {syms,n} = nothing
    
    B_deps::Union{Nothing, NamedTuple{syms, <:NTuple{n, Vector{Symbol}}}} where {syms,n} = nothing
    B_dims::Union{Nothing, NamedTuple{syms, <:NTuple{n, Vector{Int64}}}} where {syms,n} = nothing
    
    action_options::Union{Nothing, NamedTuple{syms, <:NTuple{n, Vector{Symbol}}}} where {syms,n} = nothing
    action_dims::Union{Nothing, NamedTuple{syms, <:NTuple{n, Vector{Int64}}}} where {syms,n} = nothing

    policy_iterator::Union{Nothing, NTuple{N1, NTuple{N2, NTuple{N3, Int64}}} where {N1,N2,N3}} = nothing
    action_iterator::Union{Nothing, NTuple{N1,NTuple{N2, Int64}} where {N1,N2}} = nothing
    policy_tests::Union{Nothing, NamedTuple} = nothing
    null_actions::Union{Nothing, NTuple{N,Symbol} where {N}} = nothing
    number_policies::Union{Nothing, Int64} = nothing
end

# MetaModel will be passed to agent. Every agent needs this.
@kwdef struct MetaModel
    obs_deps::Union{Nothing, NamedTuple{syms, <:NTuple{n, Vector{Symbol}}}} where {syms,n}
    obs_dims::Union{Nothing, NamedTuple{syms, <:NTuple{n, Vector{Int64}}}} where {syms,n}
    state_deps::Union{Nothing, NamedTuple{syms, <:NTuple{n, Vector{Symbol}}}} where {syms,n}
    state_dims::Union{Nothing, NamedTuple{syms, <:NTuple{n, Vector{Int64}}}} where {syms,n}
    action_options::Union{Nothing, NamedTuple{syms, <:NTuple{n, Vector{Symbol}}}} where {syms,n}
    action_dims::Union{Nothing, NamedTuple{syms, <:NTuple{n, Vector{Int64}}}} where {syms,n}
    policy_tests::Union{Nothing, NamedTuple} = nothing
    null_actions::Union{Nothing, NTuple{N,Symbol} where {N}} = nothing
    number_policies::Union{Nothing, Int64} = nothing
    
end

    
