module ActiveInference

include("maths.jl")
include("Environments\\EpistChainEnv.jl")
include("agent.jl")
include("utils.jl")
include("inference.jl")

export # maths.jl
       norm_dist,
       sample_category,
       softmax,
       spm_log_single,
       entropy,
       kl_divergence,
       get_joint_likelihood,
       dot_likelihood,
       spm_log_array_any,

       # utils.jl
       array_of_any, 
       array_of_any_zeros, 
       array_of_any_uniform, 
       onehot,
       construct_policies_full,
       plot_gridworld,
       process_observation,
       get_model_dimensions,
       to_array_of_any,


       # agent.jl
       initialize_agent,
       infer_states!,
       infer_policies!,
       sample_action!,

       # inference.jl
       get_expected_states,
       update_posterior_states,
       fixed_point_iteration,
       compute_accuracy,
       calc_free_energy


    # From Environments\\EpistChainEnv.jl
    module Environments

    include("Environments\\EpistChainEnv.jl")
    
    export EpistChainEnv, step!, reset!
       
    end
end






