using ActiveInference: AIFModel


function ActionModels.initialize_attributes(
    submodel::AIFModel,
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {TF,TI}

    #For now, we don't initialize any attributes
    return submodel

end