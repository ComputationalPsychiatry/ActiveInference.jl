using ActiveInference: AIFModel


function ActionModels.get_state_types(::AIFModel)

    return (
        posterior_states = Union{Nothing, Vector{Vector{Float64}}},
        previous_posterior_states = Union{Nothing, Vector{Vector{Float64}}},
        observation = Union{Nothing, Vector{Int64}},
        predicted_states = Union{Nothing, Vector{Vector{Vector{Vector{Float64}}}}},
        predicted_observations = Union{Nothing, Vector{Vector{Vector{Vector{Float64}}}}},
        prediction_states = Vector{Vector{Float64}},
        actions = Union{Nothing, Int64, Vector{Int64}},
        posterior_policies = Union{Nothing, Vector{Float64}},
        expected_free_energy = Union{Nothing, Vector{Float64}},
    )
end

function ActionModels.get_states(model::AIFModel, target_state::Union{String, Symbol})
    # Normalize to Symbol
    state_sym = target_state isa String ? Symbol(target_state) : target_state
    # Special case: "actions" maps to action_process.action
    if state_sym == :actions
        return getfield(model.action_process, :previous_action)
    end

    # Verify it’s a valid field name
    if !(state_sym in keys(ActionModels.get_state_types(model)))
        error("State $(state_sym) not found in AIFModel.")
    end

    # Check perceptual process
    if fieldname_in_type(typeof(model.perceptual_process), state_sym)
        return getfield(model.perceptual_process, state_sym)

    # Check action process
    elseif fieldname_in_type(typeof(model.action_process), state_sym)
        return getfield(model.action_process, state_sym)

    else
        error("State $(state_sym) not found in perceptual_process or action_process.")
    end
end

function fieldname_in_type(T::Type, name::Symbol)
    return name in fieldnames(T)
end