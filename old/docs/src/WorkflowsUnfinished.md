```@meta
EditURL = "../julia_files/WorkflowsUnfinished.jl"
```

## Workflows
This package has two main functions that can be used in a variety of workflows; `simulation` and `model fitting`.
We will here outline two different kind of workflows that can be implemented using the ActiveInference.jl package.
The first one will be a simulation workflow, where we are interested in simulating the agent's behaviour in a given environment.
Here, we might be interested in the behevaiour of a simulated active inference agent in an environment, given some specified parameters.
The second is a model fitting workflow, which is interesting for people in computational psychiatry/mathematical psychology. Here, we use observed data to fit an active inference mode and we will use a classical bayesian workflow in this regard.
See [Bayesian Workflow for Generative Modeling in Computational Psychiatry](https://www.biorxiv.org/content/10.1101/2024.02.19.581001v1)

### Simulation
In the simulation workflow, we are interested in simulating the agent's behaviour in a given environment. We might have some question wrt. behaviour expected under active inference,
or we want to figure out whether our experimental task is suitable for active inference modelling. For these purposes, we will use a simple simulation workflow:

- Decide on an environment the agent will interact with
- Create a generative model based on that environment
- Simulate the agent's behaviour in that environment
- Analyse and visualize the agent's behaviour and inferences
- Potential parameter recovery by model fitting on observed data

First, deciding on the environment entails that we have some dynamic that we are interested in from an active inference perspective - a specific research question.
Classical examples of environments are T-Mazes and Multi-Armed Bandits, that often involves some decision-making, explore-exploit and information seeking dynamics. These environments are easy to encode as POMDPs and are therefore suitable for active inference modelling.
Importantly though this can be any kind of environment that provides the active inference agent with observations, and most often will also take actions so that the agent can interact with the environment.

Based on an environment, you then create the generative model of the agent. Look under the [`Creating the POMDP Generative Model`](@ref "Creating the POMDP Generative Model") section for more information on how to do this.

You then simulate the agent's behaviour in that environment through a perception-action-learning loop, as described under the 'Simulation' section.
After this, you can analyse and visualize the agent's behaviour and inferences, and investigate what was important to the research question you had in mind.

Parameter recovery is also a possibility here, if you are interested in seeing whether the parameters you are interested in are in fact recoverable, or there is a dynamic in the agent-environment interaction, where a parameter cannot be specified but only inferred.
For an example of the latter, look up the 'As One and Many: Relating Individual and Emergent Group-Level Generative Models in Active Inference' paper, where parameters are inferred from group-level behaviour.

### Model Fitting with observed data
For

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

