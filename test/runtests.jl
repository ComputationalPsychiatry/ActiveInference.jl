using ActiveInference
using Test
using Glob

ActiveInference_path = dirname(dirname(pathof(ActiveInference)))

@testset verbose = true "all tests" begin
    test_path = ActiveInference_path * "/test/"

    @testset "quick tests" begin
        # Include quick tests similar to pre-commit tests
        include("quicktests.jl")
    end

    # List the Julia filenames in the testsuite
    filenames = glob("*.jl", test_path * "testsuite")

    # For each file
    for filename in filenames
        include(filename)
    end
end


