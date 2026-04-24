using Test
using Aqua
using JET
using ActiveInference
using JuliaFormatter: JuliaFormatter

@testset verbose = true "ActiveInference tests" begin
    @testset "Code formatting" begin
        @test JuliaFormatter.format(ActiveInference; verbose = false, overwrite = false)
    end

    @testset "Code linting" begin
        JET.test_package(ActiveInference; target_defined_modules = true)
    end

    @testset "Code quality" begin
        Aqua.test_all(
            ActiveInference;
            ambiguities = false,
            deps_compat = (check_extras = false,),
        )
    end

    # for name in (
    #     ".." # Add test names here"
    # )
    #     include("testsuite/test_$name.jl")
    # end
end
