"""
InfoStruct for tracking perceptual process configuration and learning settings.
"""

mutable struct PerceptualProcessInfo
    # Learning flags - whether each parameter type is being learned
    A_learning_enabled::Bool
    B_learning_enabled::Bool  
    D_learning_enabled::Bool
    
    # Optimization engine information
    optim_engine_name::String
    optim_engine_type::Type

    function PerceptualProcessInfo(A_learning::Union{Learn_A, Nothing}, B_learning::Union{Learn_B, Nothing}, D_learning::Union{Learn_D, Nothing}, optim_engine::Function)
        A_learning_enabled = !isnothing(A_learning)
        B_learning_enabled = !isnothing(B_learning)
        D_learning_enabled = !isnothing(D_learning)
        
        optim_engine_name = string(optim_engine)
        optim_engine_type = typeof(optim_engine)
        
        new(A_learning_enabled, B_learning_enabled, D_learning_enabled, 
            optim_engine_name, optim_engine_type)
    end
end