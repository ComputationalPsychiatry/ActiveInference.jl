
""" -------- Agent Mutable Struct -------- """

#using Format
#using Infiltrator
#using Revise

#show(stdout, "text/plain", x)
# @infiltrate; @assert false

mutable struct Current
    action::Union{Nothing, NamedTuple{<:Any, <:NTuple{N, Int64} where {N}}}
    sim_step::Int64
end

mutable struct Qo{T<:AbstractFloat}
    qo::NamedTuple{<:Any, <:NTuple{N, Vector{T}} where {N}}
end

struct State{T<:AbstractFloat}
    name::Symbol
    values::Union{UnitRange{Int64}, Vector{Int64}}
    labels::Union{Vector, NTuple{N, T2} where {N, T2}}
    B::Array{T, N} where {N}
    B_dim_names::NTuple{N, Symbol} where N
    B_dims::NTuple{N, Int64} where N
    D::Array{T, N} where {N}
    pB::Union{Nothing, Array{T, N} where {N}}
    pD::Union{Nothing, Array{T, N} where {N}}
    extra::Union{Nothing, NamedTuple}
end


struct Obs{T<:AbstractFloat}
    name::Symbol
    values::Union{UnitRange{Int64}, Vector{Int64}}
    labels::Union{Vector, NTuple{N, T2} where {N, T2}}
    A::Array{T, N} where {N}
    A_dim_names::NTuple{N, Symbol} where N
    A_dims::NTuple{N, Int64} where N
    pA::Union{Nothing, Array{T, N} where {N}}
    extra::Union{Nothing, NamedTuple}
end


struct Action
    name::Symbol
    values::Union{UnitRange{Int64}, Vector{Int64}}
    labels::Union{Vector, NTuple{N, T2} where {N, T2}}
    null_action::Union{Nothing, Symbol}
    extra::Union{Nothing, NamedTuple}
end


struct Preference{T<:AbstractFloat}
    name::Symbol
    C::Array{T, N} where {N}
    C_dim_names::NTuple{N, Symbol} where N
    C_dims::NTuple{N, Int64} where N
    extra::Union{Nothing, NamedTuple}
end


struct Policies{T<:AbstractFloat}
    policy_iterator::Union{Vector, NTuple{N, T2} where {N, T2}}
    action_iterator::Union{Vector, NTuple{N, T2} where {N, T2}}
    policy_length::Int64
    n_policies::Int64
    policy_tests::Union{Nothing, Function}
    action_tests::Union{Nothing, Function}
    earlystop_tests::Union{Nothing, Function}
    utility_reduction_fx::Union{Nothing, Function}
    info_gain_reduction_fx::Union{Nothing, Function}
    E_policies::Union{Nothing, Array{T, N} where {N}}
    E_actions::Union{Nothing, Array{T, N} where {N}}
    extra::Union{Nothing, NamedTuple}
end


struct Agent{T<:AbstractFloat}
    # todo: make all these types as exact as possible
    
    # model group
    model::@NamedTuple{
        states::NamedTuple{<:Any, <:NTuple{N, State{T}} where {N}},
        obs::NamedTuple{<:Any, <:NTuple{N, Obs{T}} where {N}},
        actions::NamedTuple{<:Any, <:NTuple{N, Action} where {N}},
        preferences::NamedTuple{<:Any, <:NTuple{N, Preference{T}} where {N}},
        policies::Policies{T}}
    
    settings::NamedTuple
    parameters::NamedTuple{<:Any, <:NTuple{N, T} where {N}}
    current::Current
    
    # belief group
    qs_prior::NamedTuple{<:Any, <:NTuple{N, Vector{T}} where {N}}  # Prior beliefs about future states after potential action, before processing observations
    qs_prev::NamedTuple{<:Any, <:NTuple{N, Vector{T}} where {N}}  # Prior beliefs from last simulation step
    qs::NamedTuple{<:Any, <:NTuple{N, Vector{T}} where {N}}  # Current beliefs about states, after processing observations
    qo::Qo  # Current beliefs about observations, just a placeholder, never written to
            
    # policy group
    q_pi_policies::Union{Nothing, Vector{Union{Missing, T}}} # Posterior beliefs over policies/actions  
    q_pi_actions::Union{Nothing, Vector{Union{Missing, T}}} # Posterior beliefs over policies/actions  
    G_policies::Union{Nothing, Vector{Union{Missing, T}}}  # Expected free energy of policies
    G_actions::Union{Nothing, Vector{Union{Missing, T}}}  # Expected free energy of actions
    utility::Union{Nothing, Matrix{Union{Missing, T}}} 
    info_gain::Union{Nothing, Matrix{Union{Missing, T}}} 
    risk::Union{Nothing, Matrix{Union{Missing, T}}} 
    ambiguity::Union{Nothing, Matrix{Union{Missing, T}}} 
    info_gain_A::Union{Nothing, Matrix{Union{Missing, T}}} 
    info_gain_B::Union{Nothing, Matrix{Union{Missing, T}}} 
    info_gain_D::Union{Nothing, Matrix{Union{Missing, T}}}
    
end

struct Agent2{T<:AbstractFloat}
    model::@NamedTuple{
    states::NamedTuple{<:Any, <:NTuple{N, State{T}} where {N}},
    obs::NamedTuple{<:Any, <:NTuple{N, Obs{T}} where {N}},
    actions::NamedTuple{<:Any, <:NTuple{N, Action} where {N}},
    preferences::NamedTuple{<:Any, <:NTuple{N, Preference{T}} where {N}},
    policies::Policies{T}}
end


#=
struct History
    qs::Vector{NamedTuple}
    q_pi::Vector{NamedTuple}
    EFE::Vector{T} where T<:AbstractFloat
    bayesian_model_averages::Vector{Any}
    SAPE::Vector{Any}
    extra::NamedTuple{<:Any, <:NTuple{N, Vector} where {N}}  # any extra info to be saved
end
=#





# graph structures

mutable struct ObsNode{T<:AbstractFloat}
    qs_next::NamedTuple{<:Any, <:NTuple{N, Vector{T}} where {N}}
    utility_updated::Union{Nothing, Missing, T}
    info_gain_updated::Union{Nothing, Missing, T}
    ambiguity_updated::Union{Nothing, Missing, T}
    risk_updated::Union{Nothing, Missing, T}
    G_updated::Union{Nothing, Missing, T}
    q_pi_updated::Union{Nothing, Missing, T}
    
    prob::Union{Nothing, Missing, T}
    prob_updated::Union{Nothing, Missing, T}
    
    observation::NamedTuple{<:Any, <:NTuple{N, T1} where {N,T1}}
    level::Int64
    policy::Union{Nothing, NamedTuple{<:Any, <:NTuple{N, T1} where {N,T1}}}
end


mutable struct ActionNode{T<:AbstractFloat}
    # qs size = number of state variables [number of categories per variable]
    qs::Union{Nothing, NamedTuple{<:Any, <:NTuple{N, Vector{T}} where {N}}} 
    qs_pi::Union{Nothing, NamedTuple{<:Any, <:NTuple{N, Vector{T}} where {N}}}  
    qo_pi::Union{Nothing, NamedTuple{<:Any, <:NTuple{N, Vector{T}} where {N}}}  
    
    utility::Union{Nothing, Missing, T}
    info_gain::Union{Nothing, Missing, T}
    ambiguity::Union{Nothing, Missing, T}
    risk::Union{Nothing, Missing, T}
    G::Union{Nothing, Missing, T}
    pruned::Bool  # are children of this ActionNode pruned out due to policy_prune_threshold 
    q_pi_children::Union{Nothing, T}  # q_pi over set of viable, non-pruned children of ObsNode
    
    utility_updated::Union{Nothing, Missing, T}
    info_gain_updated::Union{Nothing, Missing, T}
    ambiguity_updated::Union{Nothing, Missing, T}
    risk_updated::Union{Nothing, Missing, T}
    G_updated::Union{Nothing, Missing, T}
    q_pi_updated::Union{Nothing, Missing, T}
    
    observation::NamedTuple{<:Any, <:NTuple{N, T2} where {N, T2}}
    level::Int64
    policy::NamedTuple{<:Any, <:NTuple{N, T2} where {N, T2}}
end


struct EarlyStop  # ObsNode is parent
    msg::String 
    
end


struct BadPath  # ObsNode is parent
    msg::String 
end


mutable struct GraphEdge  
end




