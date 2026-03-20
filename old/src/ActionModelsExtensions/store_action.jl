using ActiveInference: AIFModel


#Store a single sampled action - used in model fitting and simulation
function ActionModels.store_action!(
    submodel::AIFModel,
    sampled_action::Union{A,Array{A}},
) where {A<:Real}
    if sampled_action isa Int
        sampled_action = [sampled_action]
    end
    submodel.action_process.previous_action = sampled_action
end


#=
### Store action overload for AIFModels
function ActionModels.store_action!(model_attributes::ModelAttributes, action::Union{A,Array{A}}) where {A<:Real}
    # If the model_attributes has a submodel that is an AIFModel, delegete to it 
    if hasproperty(model_attributes, :submodel) && model_attributes.submodel isa AIFModel
        ActionModels.store_action!(model_attributes.submodel, action)
    end

    # then store action in the model_attributes as usual
    first(model_attributes.actions).value = action
end
=#