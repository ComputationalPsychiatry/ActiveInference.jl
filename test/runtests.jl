using Test
using Aqua
using JET
using ActiveInference

@testset verbose = true "ActiveInference tests" begin
    @testset "Code linting" begin
        JET.test_package(ActiveInference; target_modules = (ActiveInference,))
    end

    @testset "Code quality" begin
        Aqua.test_all(
            ActiveInference;
            ambiguities = false,
            deps_compat = (check_extras = false,),
        )
    end

    for file_name in ("core_tests.jl")
        include("testsuite/$file_name")
    end
end
