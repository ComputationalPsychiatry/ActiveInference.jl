
function create_learning_priors(
    generative_model::GenerativeModel,
    A_learning::Union{Nothing, Learn_A},
    B_learning::Union{Nothing, Learn_B},
    D_learning::Union{Nothing, Learn_D}
)
    # Initialize priors for A, B, and D based on the learning settings
    if !isnothing(A_learning) && A_learning.prior == nothing
        A_learning.prior = deepcopy(generative_model.A) .* A_learning.concentration_parameter
    end

    if !isnothing(B_learning) && B_learning.prior == nothing
        B_learning.prior = deepcopy(generative_model.B) .* B_learning.concentration_parameter
    end

    if !isnothing(D_learning) && D_learning.prior == nothing
        D_learning.prior = deepcopy(generative_model.D) .* D_learning.concentration_parameter
    end

end