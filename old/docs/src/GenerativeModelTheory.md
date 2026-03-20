```@meta
EditURL = "../julia_files/GenerativeModelTheory.jl"
```

# The Generative Model Conceptually

The generative model is the parameters that constitute the agent's beliefs on how the hidden states of the environment generates observations based on states, and how hidden underlying states changes over time.
In the generative model is also the beliefs of how the agent through actions can influence the states of the environment. Together this holds the buidling blocks that allows for the perception-action loop.

There are five main buidling blocks of the generative model which are; **A**, **B**, **C**, **D**, and **E**.
Each of these contain parameters that describe the agent's beliefs about the environment.
We will now go through these conecptually one at a time.

## A
**A** is the observation likelihood model, and describes the agent's beliefs about how the hidden states of the environment generates observations.
Practically in this package, and other POMDP implemantations as well, this is described through a series of categorical distributions, meaning that for each observation, there is a categorical probability distribution over how likely each hidden state is to generate that observation.
Let us for example imagine a simple case, where the agent is in a four location state environment, could be a 2x2 gridworld. In this case, there would be one obseration linked to each hidden state, and **A** then maps the agent's belief of how likely each hidden location state is to generate each observation.
The agent can then use this belief to infer what state it is in based on the observation it receives. Let's look at an example **A**, which in this case would be a 4x4 matrix:

```math
A =
\overset{\text{\normalsize States}\vphantom{\begin{array}{c} 0 \\ 0 \end{array}}}{
    \begin{array}{cccc}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & 1
    \end{array}
}
\quad
\text{\normalsize Observations}
```

In this case, the agent is quite certain about which states produces which observations. This matrix could be made more uncertain to the point of complete uniformity and it could be made certain in the sense of each column being a one-hot vector.
In the case of a certain **A**, the generative model stops being a "partially observable" Markov decision process, and becomes a fully observable one, making it a Markov decision process (MDP). For a more technical and mathematical definition of the observation likelihood model.

## B
**B** is the transition likelihood model that encodes the agent's beliefs about how the hidden states of the environment changes over time.
This is also made up of categorical distributions, though instead of observations to states, it maps states to states.
If we take the same case again, a 2x2 gridworld, we would have a 4x4 matrix that describes how the agent believes the states evolve over time.
An extra addition to **B**, is that it can depend on actions, meaning that it can believe that the hidden states of the environment change differently depending on the action taken by the agent.
Due to this fact, we would the have a matrix for each action, making **B** a 3 dimensional tensor, with 2 dimensions for the "from" state and the "to" state, and then an action dimension.
Let's look at an example of a slice of **B** for the action "down" in the grid world, which in this case would be a 4x4 matrix:

```math
B("down") =
\overset{\text{\normalsize Previous State}\vphantom{\begin{array}{c} 0 \\ 0 \end{array}}}{
    \begin{array}{cccc}
        0 & 0 & 0 & 0 \\
        1 & 1 & 0 & 0 \\
        0 & 0 & 0 & 0 \\
        0 & 0 & 1 & 1
    \end{array}
}
\quad
\text{\normalsize Current State}
```

We could make 3 more similar matrices for the actions "up", "left", and "right", and then we would have the full **B** tensor for the gridworld. But here, the main point is that
**B** decsribes the agent's belief of how hidden states change over time, and this can be dependent on actions, but might also be independent of actions, and thus the agent believes that the changes are out of its control.

## C
**C** is the prior over observations, also called preferences over observations. This is an integral part of the utility of certain observations, i.e. it encodes how much the agent prefers or dislikes certain observations.
**C** is a simple vector over observations, where each entry is a value that describes the utility or preference of that specific observation.
If we continue with the simple 2x2 gridworld example, we would have 4 observations, one for each location state (same amount of observations as in **A**).
Let's say that we would like for the agent to dislike observing the top left location (indexed as 1), and prefer the bottom right location (indexed as 4). We would then create **C** in the following way:

```math
C =
\begin{array}{cccc}
    -2 & 0 & 0 & 2 \\
\end{array}
```

The magnitude of the values in **C** is arbitrary, and denotes a ratio and amount of dislike/preference. Here, we have chosen the value of -2 and 2
to encode that the agent dislikes the top left location just as much as it likes the bottom right location. The zeros in between just means that the agent has not preference or dislike for these locatin observations.
Note that since **C** is not a categorical distribution, it does not need to sum to 1, and the values can be any real number.

## D
**D** is the prior over states, and is the agent's beliefs about the initial state of the environment. This is also a simple vector that is a categorical distribution.
Note that if **A** is certain, then **D** does not matter a lot for the inference process, as the agent can infer the state from the observation. However, if **A** is uncertain,
then **D** becomes very important, as it serves as the agent's anchor point of where it is initially in the environment. In the case of out
2x2 gridworld, we would have a vector with 4 entries, one for each location state. If we assume that the agent correctly infers it's initial location as upper left corner, **D** would look like:

```math
D =
\begin{array}{cccc}
    1 & 0 & 0 & 0 \\
\end{array}
```

## E
**E** is the prior over policies, and can be described as the agent's habits. Policies in Active Inference vernacular are sets of actions, with an action for each step in the future, specified by a policy length.
It is a categorical distribution over policies, with a probability for each policy. This will have an effect on the agent posterior over policies,
which is the probability of taking a certain action at a time step. This will often be set to a uniform distribution, if we are not interested in giving the agent habits.
Let us assume that we will give our agent a uniform **E** for a policy length of 2, this mean that we will have a uniform categorical distribution over 16 possible policies ``(4 (actions) ^ {2 (policy length)})``:

```math
E =
\begin{array}{cccc}
0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 \\
\end{array}
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

