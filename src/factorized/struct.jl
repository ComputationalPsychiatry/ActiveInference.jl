
""" -------- Agent Mutable Struct -------- """

#using Format
#using Infiltrator
#using Revise

#show(stdout, "text/plain", x)
# @infiltrate; @assert false


mutable struct Agent
    # todo: make all these types as exact as possible
    
    # model group
    model::NamedTuple
    parameters::NamedTuple
    settings::NamedTuple
    
    # belief group
    last_action::Union{Nothing, NamedTuple} 
    qs_prior::NamedTuple{<:Any, <:NTuple{N, Vector{Float64}} where {N}}  # Prior beliefs about states after action, before processing observations
    qs_current::NamedTuple{<:Any, <:NTuple{N, Vector{Float64}} where {N}}  # Current beliefs about states, after processing observations
    qo_current::NamedTuple{<:Any, <:NTuple{N, Vector{Float64}} where {N}}  # Current beliefs about observations
            
    # policy group
    q_pi::Vector{Union{Missing, T}} where T <:Real # Posterior beliefs over policies
    G::Vector{Union{Missing, T}} where T <:Real  # Expected free energy of policies
    utility::Union{Nothing, Matrix{Union{Missing, Float64}}} 
    info_gain::Union{Nothing, Matrix{Union{Missing, Float64}}} 
    risk::Union{Nothing, Matrix{Union{Missing, Float64}}} 
    ambiguity::Union{Nothing, Matrix{Union{Missing, Float64}}} 
    info_gain_A::Union{Nothing, Matrix{Union{Missing, Float64}}} 
    info_gain_B::Union{Nothing, Matrix{Union{Missing, Float64}}} 
    info_gain_D::Union{Nothing, Matrix{Union{Missing, Float64}}}   
    history::NamedTuple
end



# graph structures

mutable struct ObsNode
    qs_next::NamedTuple{<:Any, <:NTuple{N, Vector{Float64}} where {N}}
    utility_updated::Union{Missing, Nothing, Float64}
    info_gain_updated::Union{Missing, Nothing, Float64}
    ambiguity_updated::Union{Missing, Nothing, Float64}
    risk_updated::Union{Missing, Nothing, Float64}
    G_updated::Union{Missing, Nothing, Float64}
    q_pi_updated::Union{Missing, Nothing, Float64}
    
    prob::Union{Missing, Nothing, Float64}
    prob_updated::Union{Missing, Nothing, Float64}
    
    #subpolicy::Union{Missing, Nothing, Tuple}
    observation::NamedTuple{<:Any, <:NTuple{N, T} where {N, T}}
    level::Int64
    policy::Union{Nothing, NamedTuple{<:Any, <:NTuple{N, T} where {N, T}}}
end


mutable struct ActionNode
    # qs size = number of state variables [number of categories per variable]
    qs::Union{Nothing, NamedTuple{<:Any, <:NTuple{N, Vector{Float64}} where {N}}} 

    # qs_pi size = number of actions (=1) [number of state variables [ number of categories per variable]]
    qs_pi::Union{Nothing, NamedTuple{<:Any, <:NTuple{N, Vector{Float64}} where {N}}}  

    # qo_pi size  = number of actions (=1) [number of observation variables [ number of categories per variable]]
    qo_pi::Union{Nothing, NamedTuple{<:Any, <:NTuple{N, Vector{Float64}} where {N}}}  
    
    utility::Union{Missing, Nothing, Float64}
    info_gain::Union{Missing, Nothing, Float64}
    ambiguity::Union{Missing, Nothing, Float64}
    risk::Union{Missing, Nothing, Float64}
    G::Union{Missing, Nothing, Float64}
    pruned::Bool  # are children of this ActionNode pruned out due to policy_prune_threshold 
    q_pi_children::Union{Nothing, Float64}  # q_pi over set of viable, non-pruned children of ObsNode
    #G_children::Union{Nothing, Float64}  # G over set of viable children of ObsNode, including any pruning penalty
    
    utility_updated::Union{Missing, Nothing, Float64}
    info_gain_updated::Union{Missing, Nothing, Float64}
    ambiguity_updated::Union{Missing, Nothing, Float64}
    risk_updated::Union{Missing, Nothing, Float64}
    G_updated::Union{Missing, Nothing, Float64}
    q_pi_updated::Union{Missing, Nothing, Float64}
    
    observation::NamedTuple{<:Any, <:NTuple{N, T} where {N, T}}
    level::Int64
    policy::NamedTuple{<:Any, <:NTuple{N, T} where {N, T}}
end


struct EarlyStop  # ObsNode is parent
    msg::String 
    
end


struct BadPath  # ObsNode is parent
    msg::String 
end


mutable struct GraphEdge  
end


struct Label
    level::Int64
    observation::Union{Nothing, NTuple{N, Int64} where N}
    action::Union{Nothing, NTuple{N, Int64}} where {N}
    type::String
end


struct Actions
    level::Int64
    observation::Union{Nothing, NTuple{N, Int64} where N}
    action::Union{Nothing, NTuple{N, Int64}} where {N}
    type::String
end


