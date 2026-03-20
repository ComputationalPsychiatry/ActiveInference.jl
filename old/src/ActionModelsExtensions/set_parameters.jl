using ActiveInference: AIFModel


### Setting a Single Parameter
function ActionModels.set_parameters!(
    model::AIFModel,
    target_param::Symbol,
    param_value::Any
)
    @show target_param
    @show param_value
    if target_param == :learning_rate_A
        model.perceptual_process.A_learning.learning_rate = param_value
    elseif target_param == :forgetting_rate_A
        model.perceptual_process.A_learning.forgetting_rate = param_value
    elseif target_param == :learning_rate_B
        model.perceptual_process.B_learning.learning_rate = param_value
    elseif target_param == :forgetting_rate_B
        model.perceptual_process.B_learning.forgetting_rate = param_value
    elseif target_param == :gamma
        model.action_process.gamma = param_value
    else
        @warn "Unknown parameter: $target_param"
    end
end