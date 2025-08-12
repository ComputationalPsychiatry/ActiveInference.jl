"""
InfoStruct for tracking perceptual process configuration and learning settings.
"""

struct CAVIInfo

    # Learning flag - whether any parameters are being learned
    learning_enabled::Bool

    # Learning flags - whether each parameter type is being learned
    A_learning_enabled::Bool
    B_learning_enabled::Bool  
    D_learning_enabled::Bool
    
    # Optimization engine information
    perceptual_process_name::String

    function CAVIInfo(A_learning::Union{Learn_A, Nothing}, B_learning::Union{Learn_B, Nothing}, D_learning::Union{Learn_D, Nothing})
        
        # Check if any learning is enabled
        learning_enabled = !isnothing(A_learning) || !isnothing(B_learning) || !isnothing(D_learning)

        A_learning_enabled = !isnothing(A_learning)
        B_learning_enabled = !isnothing(B_learning)
        D_learning_enabled = !isnothing(D_learning)
        
        # Perceptual Process
        perceptual_process_name = "CAVI"

        new(learning_enabled, A_learning_enabled, B_learning_enabled, D_learning_enabled, 
            perceptual_process_name)
    end
end

"""
Pretty print function for CAVIInfo.
"""
function show_info(info::CAVIInfo; verbose::Bool = true)
    if !verbose
        return
    end
    
    println("\n" * "="^100)
    println("👁️  Perceptual Process Information")
    println("="^100)

    clean_process_name = replace(string(info.perceptual_process_name), r"ActiveInference\.DiscretePOMDP\.NonFactorized\." => "")

    println("\n⚙️  Perceptual Process: $clean_process_name")
    
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