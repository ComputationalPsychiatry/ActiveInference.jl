```@meta
EditURL = "../julia_files/Introduction.jl"
```

# Introduction to the ActiveInference.jl package

This package is a Julia implementation of the Active Inference framework, with a specific focus on cognitive modelling.
In its current implementation, the package is designed to handle scenarios that can be modelled as discrete state spaces, with 'partially observable Markov decision process' (POMDP).
In this documentation we will go through the basic concepts of how to use the package for different purposes; simulation and model inversion with Active Inference, also known as parameter estimation.

## Installing Package
Installing the package is done by adding the package from the julia official package registry in the following way:

```julia
using Pkg
Pkg.add("ActiveInference")
```

Now, having added the package, we simply import the package to start using it:

```julia
using ActiveInference
```

In the next section we will go over the basic concepts of how to start using the package. We do this by providing instructions on how to create and design a generative model, that can be used for both simulation and parameter estimation.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

