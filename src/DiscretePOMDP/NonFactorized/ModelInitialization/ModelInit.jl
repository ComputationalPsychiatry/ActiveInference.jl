""" Constructor for the AIFModel in the DiscretePOMDP module."""
function ActiveInferenceCore.AIFModel(;
    generative_model::GenerativeModel,
    perceptual_process::PP,
    action_process::ActionProcess
) where PP <: AbstractPerceptualProcess
    
    fill_missing_parameters(generative_model, perceptual_process, action_process)

    return AIFModel(generative_model, perceptual_process, action_process)
end