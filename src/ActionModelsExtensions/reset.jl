"""
Resets an AIF type agent to its initial state

    reset!(aif::AIF)

"""

using ActionModels

function ActionModels.reset!(aif::POMDPActiveInference)
    # Reset the agent's state fields to initial conditions
    aif.states.qs_current = create_matrix_templates([size(aif.parameters.B[f], 1) for f in eachindex(aif.parameters.B)])
    aif.states.prior = aif.parameters.D
    aif.states.q_pi = ones(length(aif.settings.policies)) / length(aif.settings.policies)
    aif.states.G = zeros(length(aif.settings.policies))
    aif.states.action = Int[]

    # Clear the history in the states dictionary
    aif.history = construct_history_struct(aif.states)
    return nothing
end