
# include("./grid.jl")
#show(stdout, "text/plain", x)
# @infiltrate; @assert false

include("./structs.jl")

using Format
using Infiltrator
using Revise



# --------------------------------------------------------------------------------------------------
function get_A_dimensions(model)
    """
    Get dimensions, deps, and dependencies for A matrices. 
    """

    observations = model.observations
    states = model.states

    A_deps = NamedTuple()
    A_dims = NamedTuple()
    
    Deps = Vector{Vector{Symbol}}()
    Dims = Vector{Vector{Int64}}()
    
    for (ii, fname) in enumerate(fieldnames(Observations))  # fname is Symbol
        observation = getfield(observations, fname)
        deps = [observation.name]
        @assert !isnothing(observation.n)

        dims = [observation.n]
        
        # observation dependencies
        for sname in observation.dependencies
            obj = getfield(states, Symbol(sname))
            push!(deps, obj.name)
            @assert !isnothing(obj.n)
            push!(dims, obj.n)
        end
        push!(Deps, deps)
        push!(Dims, dims)
    end

    A_deps = NamedTuple{fieldnames(Observations)}(Deps)
    A_dims = NamedTuple{fieldnames(Observations)}(Dims)
    
    model.A_deps = A_deps
    model.A_dims = A_dims
    
end


# --------------------------------------------------------------------------------------------------
function get_B_dimensions(model)
    """
    Get dimensions, deps, and dependencies for B matrices. 
    Each matrix is of dimensions [state_dims, dependency_dims..., actions_dims...]
    """
    
    states = model.states
    actions = model.actions
    
    B_deps = NamedTuple()
    B_dims = NamedTuple()
    
    Deps = Vector{Vector{Symbol}}()
    Dims = Vector{Vector{Int64}}()
    
    for (ii, fname) in enumerate(fieldnames(States))  # fname is Symbol
        state = getfield(states, fname)
        
        deps = [state.name]
        @assert !isnothing(state.n)

        dims = [state.n]
        
        # state dependencies
        for sname in state.dependencies
            obj = getfield(states, Symbol(sname))
            push!(deps, obj.name)
            @assert !isnothing(obj.n)
            push!(dims, obj.n)
        end

        # action dependencies
        # this does not add an extra dim to the end of the matrix if no action
        for sname in state.action_dependencies
            obj = getfield(actions, Symbol(sname))
            push!(deps, obj.name)
            @assert !isnothing(obj.n)
            push!(dims, obj.n)
        end
        push!(Deps, deps)
        push!(Dims, dims)

    end
    
    B_deps = NamedTuple{fieldnames(States)}(Deps)
    B_dims = NamedTuple{fieldnames(States)}(Dims)
    
    model.B_deps = B_deps
    model.B_dims = B_dims

    #@infiltrate; @assert false
end


# --------------------------------------------------------------------------------------------------
function get_action_dimensions(model)
    
    actions = model.actions
    
    action_deps = NamedTuple()
    action_dims = NamedTuple()
    
    Options = Vector{Vector{Symbol}}()
    Dims = Vector{Vector{Int64}}()
    
    for (ii, fname) in enumerate(fieldnames(Actions))  # fname is Symbol
        action = getfield(actions, fname)
        options = action.labels  
        @assert !isnothing(action.n)
        dims = [action.n]
        push!(Options, options)
        push!(Dims, dims)

    end
    
    action_options = NamedTuple{fieldnames(Actions)}(Options)
    action_dims = NamedTuple{fieldnames(Actions)}(Dims)
    
    model.action_options = action_options
    model.action_dims = action_dims

    #@infiltrate; @assert false
end

