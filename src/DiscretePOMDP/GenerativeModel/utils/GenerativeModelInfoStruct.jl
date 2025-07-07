mutable struct InfoStruct

    model_type::String
    abstract_model_type::String
    components::Vector{String}

    n_modalities::Int
    n_factors::Int
    n_states::Vector{Int}
    n_observations::Vector{Int}

    controllable_factors::Vector{Int}
    controls_per_factor::Vector{Int}
    
    """
    Internal constructor for InfoStruct that extracts information from generative model parameters.
    """
    function InfoStruct(A, B, C, D)

        # Determine which components are present (should all be, but in case something is missing this is useful)
        components = String[]
        if !isnothing(A)
            push!(components, "A")
        end
        if !isnothing(B)
            push!(components, "B")
        end
        if !isnothing(C)
            push!(components, "C")
        end
        if !isnothing(D)
            push!(components, "D")
        end
        
        # Extract dimensional information from A matrix
        n_modalities = length(A)
        n_observations = [size(A_m, 1) for A_m in A]
        
        # Infer number of factors and states from first A matrix
        A_dims = size(A[1])
        n_factors = length(A_dims) - 1  # Here, first dimension is observations
        n_states = [A_dims[i+1] for i in 1:n_factors]

        # Extract control information from B matrices
        controls_per_factor = [size(B_f, 3) for B_f in B]
        controllable_factors = [i for (i, n_controls) in enumerate(controls_per_factor) if n_controls > 1]
        
        # Extract the complete type string from the GenerativeModel type
        model_type = string(GenerativeModel)

        # Extract the supertype
        abstract_model_type = string(supertype(GenerativeModel))
        
        return new(
            model_type,
            abstract_model_type,
            components,
            n_modalities,
            n_factors,
            n_states,
            n_observations,
            controllable_factors,
            controls_per_factor
        )
    end
end

"""
Pretty print function for InfoStruct.
"""
function show_info(info::InfoStruct; verbose::Bool = true)
    if !verbose
        return
    end
    
    println("\n" * "="^100)
    println("🧠 Generative Model Information")
    println("="^100)

    clean_model_type = replace(info.model_type, r"ActiveInference.ActiveInferenceCore." => "")
    clean_abstract_type = replace(info.abstract_model_type, r"ActiveInference.ActiveInferenceCore." => "")
    
    println("📋 Model Type:$(" "^9) $clean_model_type")
    println("🔍 Abstract Model Type: $clean_abstract_type")
    println("🧩 Components:$(" "^9) $(join(info.components, ", "))")
    
    if info.n_modalities > 0
        println("\n📊 Structure:")
        println("   • Observation modalities: $(info.n_modalities)")
        println("   • State factors: $(info.n_factors)")
        println("   • States per factor: $(info.n_states)")
        println("   • Observations per modality: $(info.n_observations)")
    end
    
    if !isempty(info.controls_per_factor)
        println("\n🎮 Control Structure:")
        println("   • Controls per factor: $(info.controls_per_factor)")
        if !isempty(info.controllable_factors)
            println("   • Controllable factors: $(info.controllable_factors)")
        else
            println("   • No controllable factors")
        end
    end
    
    println("="^100)
end