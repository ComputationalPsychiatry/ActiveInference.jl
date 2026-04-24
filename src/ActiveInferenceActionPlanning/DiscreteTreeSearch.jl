# Doing the full tree-search 
# discrete observations 
# outputs discrete actions 

# Make the tree -> go through it 
# 1. Full unpruned trea

# 2- Make it sophisticated from the beginning, allow non-sophisticated to be a special case 

# 3. allow different pruning methods 

# 4. Get good at Meta-graphs.jl -> do we need it? or can we just make our own tree? 

# 5.:

# Different ways we can make the "planning"
# - sophisticated / non-sophisticated

# - Which objective to use? 
#    - EFE => and all alternatives (full EFE, different subtypes, different full EFE alternatives)
#    - Inductive Inference / or not 

# - Tree-search methods:
#   - Monte-Carlo Tree Search (MCTS)
#   - Explicit tree search
#   - recursive - implicit tree search 

# - Pruning methods:
#  - remove actions with low probabilities 
#  - cut observations with low probabilities 

# Rule-based filtering:
# - a-priori action-combinations that are not allowed 
# - we also have state-dependent action-combinations that are not allowed 

# 6. 
# Go into pymdp and check what they do, prioritze different parts of this list <-- do this first 

# also look at Monte-Carlo tree search package in Julia -> will just extend that 

# 7. 
# look into other tree-search methods (in Julia or so)

# 8. 
# In the explicit graph, what intermediate values do we store for inspection?? 

# 9. What exactly needs to be done on every node? -> check!!!





