function interaction_loop(model::AIFModel, environment, T)

    previous_action = missing
    environment_belief = missing
    observation = missing

    for t in 1:T
    
        action_posterior, environment_belief, infer_environment_snapshot, infer_actions_snapshot = active_inference!(model, observation, previous_action, environment_belief)

        previous_action = rand(action_posterior)

        observation = run_environment(environment, previous_action)

        push!(history, (observation, previous_action, environment_belief, action_posterior, infer_environment_snapshot, infer_actions_snapshot))
    end

end