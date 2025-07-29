"""
Resets an Agent type agent to its initial state

    reset!(agent::Agent)

"""

using ActionModels

function ActionModels.reset!(agent::Agent)
    # Reset the agent's state fields to initial conditions
    agent.qs = create_matrix_templates([size(agent.B[f], 1) for f in eachindex(agent.B)])
    agent.prior = agent.D
    agent.Q_pi = ones(length(agent.policies)) / length(agent.policies)
    agent.G = zeros(length(agent.policies))
    agent.action = Int[]

    # Clear the history in the states dictionary
    for key in keys(agent.states)

        if key != "policies"
            agent.states[key] = []
        end
    end
    return nothing
end