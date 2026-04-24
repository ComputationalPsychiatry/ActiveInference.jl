# Tests for the ActiveInferenceCore module.
#
# Covers:
#   - Abstract type hierarchies (action types, observation types, inference markers)
#   - AbstractGenerativeModel parametric type constraints
#   - AIFModel struct: valid construction, field access, error constructor
#   - Placeholder functions: infer_environment, infer_actions, set_variables!
#   - active_inference! pipeline: error path and full integration with mock dispatch

# ---------------------------------------------------------------------------
# Concrete mock types used across multiple test sets
# ---------------------------------------------------------------------------

struct _MockGenerativeModel <:
       AbstractGenerativeModel{DiscreteActions, DiscreteObservations} end
struct _MockInferenceEnvironment <: AbstractInferenceEnvironment end
struct _MockInferenceActions <: AbstractInferenceActions end

# ---------------------------------------------------------------------------
# Concrete types for the integration test (distinct from the mocks above so
# method dispatch remains unambiguous).
# ---------------------------------------------------------------------------

struct _ConcreteGenerativeModel <:
       AbstractGenerativeModel{ContinuousActions, ContinuousObservations} end
struct _ConcreteInferenceEnvironment <: AbstractInferenceEnvironment end
struct _ConcreteInferenceActions <: AbstractInferenceActions end

# Full pipeline implementations for the integration model
function infer_environment(
    ::AIFModel{
        _ConcreteGenerativeModel,
        _ConcreteInferenceEnvironment,
        _ConcreteInferenceActions,
    },
    observation,
    previous_action,
)
    return observation + 1
end

function infer_actions(
    ::AIFModel{
        _ConcreteGenerativeModel,
        _ConcreteInferenceEnvironment,
        _ConcreteInferenceActions,
    },
    environment_posterior,
)
    return environment_posterior * 2
end

function set_variables!(
    model::AIFModel{
        _ConcreteGenerativeModel,
        _ConcreteInferenceEnvironment,
        _ConcreteInferenceActions,
    },
    observation,
    previous_action,
    environment_posterior,
    action_posterior,
)
    return model
end

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@testset "ActiveInferenceCore" begin

    # --- Action type hierarchy -------------------------------------------
    @testset "Action type hierarchy" begin
        @test isabstracttype(AbstractActionType)

        # All four subtypes are concrete structs
        @test isconcretetype(DiscreteActions)
        @test isconcretetype(ContinuousActions)
        @test isconcretetype(MixedActions)
        @test isconcretetype(NoActions)

        # Each concrete type is still a subtype of the abstract parent
        @test DiscreteActions <: AbstractActionType
        @test ContinuousActions <: AbstractActionType
        @test MixedActions <: AbstractActionType
        @test NoActions <: AbstractActionType

        # Concrete singleton structs can be instantiated
        @test DiscreteActions() isa AbstractActionType
        @test ContinuousActions() isa AbstractActionType
        @test MixedActions() isa AbstractActionType
        @test NoActions() isa AbstractActionType

        # Siblings must not be subtypes of each other
        @test !(ContinuousActions <: DiscreteActions)
        @test !(DiscreteActions <: ContinuousActions)
        @test !(MixedActions <: NoActions)
    end

    # --- Observation type hierarchy ---------------------------------------
    @testset "Observation type hierarchy" begin
        @test isabstracttype(AbstractObservationType)

        # All four subtypes are concrete structs
        @test isconcretetype(DiscreteObservations)
        @test isconcretetype(ContinuousObservations)
        @test isconcretetype(MixedObservations)
        @test isconcretetype(NoObservations)

        # Each concrete type is still a subtype of the abstract parent
        @test DiscreteObservations <: AbstractObservationType
        @test ContinuousObservations <: AbstractObservationType
        @test MixedObservations <: AbstractObservationType
        @test NoObservations <: AbstractObservationType

        # Concrete singleton structs can be instantiated
        @test DiscreteObservations() isa AbstractObservationType
        @test ContinuousObservations() isa AbstractObservationType
        @test MixedObservations() isa AbstractObservationType
        @test NoObservations() isa AbstractObservationType

        # No cross-hierarchy subtyping
        @test !(DiscreteObservations <: AbstractActionType)
        @test !(DiscreteActions <: AbstractObservationType)
    end

    # --- AbstractGenerativeModel -----------------------------------------
    @testset "AbstractGenerativeModel" begin
        @test isabstracttype(AbstractGenerativeModel)

        # Parameterised specialisations with concrete marker types are valid subtypes
        @test AbstractGenerativeModel{DiscreteActions, DiscreteObservations} <:
              AbstractGenerativeModel
        @test AbstractGenerativeModel{ContinuousActions, ContinuousObservations} <:
              AbstractGenerativeModel
        @test AbstractGenerativeModel{MixedActions, MixedObservations} <:
              AbstractGenerativeModel
        @test AbstractGenerativeModel{NoActions, NoObservations} <: AbstractGenerativeModel

        # User-defined concrete subtype satisfies the hierarchy
        @test _MockGenerativeModel <: AbstractGenerativeModel
        @test _MockGenerativeModel <:
              AbstractGenerativeModel{DiscreteActions, DiscreteObservations}
    end

    # --- AbstractInferenceEnvironment / AbstractInferenceActions ----------
    @testset "Inference marker abstract types" begin
        @test isabstracttype(AbstractInferenceEnvironment)
        @test isabstracttype(AbstractInferenceActions)

        @test _MockInferenceEnvironment <: AbstractInferenceEnvironment
        @test _MockInferenceActions <: AbstractInferenceActions

        # No cross-subtyping
        @test !(_MockInferenceEnvironment <: AbstractInferenceActions)
        @test !(_MockInferenceActions <: AbstractInferenceEnvironment)
    end

    # --- AIFModel struct --------------------------------------------------
    @testset "AIFModel struct" begin

        @testset "Valid construction via inner constructor" begin
            gm = _MockGenerativeModel()
            ie = _MockInferenceEnvironment()
            ia = _MockInferenceActions()

            model = AIFModel{
                _MockGenerativeModel,
                _MockInferenceEnvironment,
                _MockInferenceActions,
            }(gm, ie, ia)

            @test model isa AIFModel
            @test model isa AIFModel{
                _MockGenerativeModel,
                _MockInferenceEnvironment,
                _MockInferenceActions,
            }
        end

        @testset "Field access" begin
            gm = _MockGenerativeModel()
            ie = _MockInferenceEnvironment()
            ia = _MockInferenceActions()

            model = AIFModel{
                _MockGenerativeModel,
                _MockInferenceEnvironment,
                _MockInferenceActions,
            }(gm, ie, ia)

            @test model.generative_model === gm
            @test model.inference_environment === ie
            @test model.inference_actions === ia

            @test model.generative_model isa AbstractGenerativeModel
            @test model.inference_environment isa AbstractInferenceEnvironment
            @test model.inference_actions isa AbstractInferenceActions
        end

        @testset "Error constructor rejects wrong types" begin
            @test_throws ArgumentError AIFModel("not_a_model", "not_env", "not_action")
            @test_throws ArgumentError AIFModel(42, nothing, nothing)
        end
    end

    # --- Placeholder functions throw ArgumentError -----------------------
    @testset "Placeholder functions" begin
        model = AIFModel{
            _MockGenerativeModel,
            _MockInferenceEnvironment,
            _MockInferenceActions,
        }(_MockGenerativeModel(), _MockInferenceEnvironment(), _MockInferenceActions())

        @testset "infer_environment throws ArgumentError" begin
            err = @test_throws ArgumentError infer_environment(model, nothing, nothing)
            @test occursin("infer_environment", err.value.msg)
        end

        @testset "infer_actions throws ArgumentError" begin
            err = @test_throws ArgumentError infer_actions(model, nothing)
            @test occursin("infer_actions", err.value.msg)
        end

        @testset "set_variables! throws ArgumentError" begin
            err = @test_throws ArgumentError set_variables!(
                model, nothing, nothing, nothing, nothing
            )
            @test occursin("set_variables!", err.value.msg)
        end

        @testset "active_inference! propagates ArgumentError from infer_environment" begin
            @test_throws ArgumentError active_inference!(model, nothing, nothing)
        end
    end

    # --- Integration: full pipeline with concrete dispatch ---------------
    @testset "Integration: active_inference! with concrete dispatch" begin
        model = AIFModel{
            _ConcreteGenerativeModel,
            _ConcreteInferenceEnvironment,
            _ConcreteInferenceActions,
        }(
            _ConcreteGenerativeModel(),
            _ConcreteInferenceEnvironment(),
            _ConcreteInferenceActions(),
        )

        @testset "infer_environment returns observation + 1" begin
            @test infer_environment(model, 3, 0) == 4
            @test infer_environment(model, 0, 0) == 1
            @test infer_environment(model, -5, 99) == -4
        end

        @testset "infer_actions returns environment_posterior * 2" begin
            @test infer_actions(model, 6) == 12
            @test infer_actions(model, 0) == 0
            @test infer_actions(model, -3) == -6
        end

        @testset "active_inference! full pipeline" begin
            # With observation=5: env_posterior = 6, action_posterior = 12
            @test active_inference!(model, 5, 0) == 12
            @test active_inference!(model, 0, 0) == 2
            @test active_inference!(model, -1, 0) == 0
        end

        @testset "active_inference! returns action posterior" begin
            result = active_inference!(model, 10, 0)
            @test result isa Number
            @test result == 22  # (10 + 1) * 2
        end
    end
end;
