"""
InfoStruct for tracking perceptual process configuration and learning settings.
"""

struct PerceptualProcessInfo

    # Learning flag - whether any parameters are being learned
    learning_enabled::Bool

    # Learning flags - whether each parameter type is being learned
    A_learning_enabled::Bool
    B_learning_enabled::Bool  
    D_learning_enabled::Bool
    
    # Optimization engine information
    optim_engine_name::String
    optim_engine_type::Type

    function PerceptualProcessInfo(A_learning::Union{Learn_A, Nothing}, B_learning::Union{Learn_B, Nothing}, D_learning::Union{Learn_D, Nothing}, optim_engine::Union{AbstractOptimEngine, Missing})
        
        # Check if any learning is enabled
        learning_enabled = !isnothing(A_learning) || !isnothing(B_learning) || !isnothing(D_learning)

        A_learning_enabled = !isnothing(A_learning)
        B_learning_enabled = !isnothing(B_learning)
        D_learning_enabled = !isnothing(D_learning)
        
        # Handle missing optim_engine
        if optim_engine === missing
            optim_engine_name = "Missing"
            optim_engine_type = Missing
        else
            optim_engine_name = string(optim_engine)
            optim_engine_type = typeof(optim_engine)
        end
        
        new(learning_enabled, A_learning_enabled, B_learning_enabled, D_learning_enabled, 
            optim_engine_name, optim_engine_type)
    end
end

"""
Pretty print function for PerceptualProcessInfo.
"""
function show_info(info::PerceptualProcessInfo; verbose::Bool = true)
    if !verbose
        return
    end
    
    println("\n" * "="^100)
    println("👁️  Perceptual Process Information")
    println("="^100)

    clean_engine_type = replace(string(info.optim_engine_type), r"ActiveInference\.DiscretePOMDP\.NonFactorized\." => "")
    
    println("\n⚙️  Optimization Engine: $clean_engine_type")
    
    println("\n📊 Learning Configuration:")

    if !info.learning_enabled
        println("   • Learning enabled: $(info.learning_enabled)")
    end

    if info.learning_enabled
        println("   • A-parameter learning: $(info.A_learning_enabled)")
        println("   • B-parameter learning: $(info.B_learning_enabled)")
        println("   • D-parameter learning: $(info.D_learning_enabled)")
    end
    
    println("="^100)
end