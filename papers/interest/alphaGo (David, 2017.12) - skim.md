**Title: Mastering the game of Go with deep neural networks and tree search**

# What did the authors tried to accomplished?

Use NN to solve Go game.  
- Developed effective **move selection** and **position evaluation** functions for Go.
- novel combination of **supervised** and **reinforcement learning**.
- introduced a new search algorithm that successfully combines **neural network evaluations** with **Monte Carlo rollouts**
- Our program AlphaGo integrates these components together, **at scale**, in a high-performance **tree search engine**
  
# Key elements of the approach

1. a supervised learning (**SL**) policy network to learn from **human expert plays** (in a data set of positions)
2. train a **fast policy** that can rapidly **sample actions** during rollouts   
(faster but less accurate)
3. train a reinforcement learning (**RL**) policy network that improves the SL policy network by optimizing the final outcome of games of **self play**.   
(maximize the outcome (that is, winning more games) against previous versions of the policy network, whcih also generates new data set)
4. train a **value network** that **predicts the winner** of games played by the RL policy network against itself (by regression)  
(that is, whether sampled state-action pairs (s, a), using stochastic gradient ascent to maximize the likelihood of the human move a selected in state s)

# Takeaway

- Refing SL learnt gradient with RL gives better result.   
- Monte Carlo sampling can reduce search space.

# Other references to follow

# Results (Good or Bad)

**fast policy.** 2 μs to select an action, rather than 3 ms for the policy network.

**go deeper.** 10%+ than previous SOTA.

**RL refine SL.** RL policy network won more than 80% of games against the SL policy network

**Why it matters?** The game of Go has long been viewed as the most challenging of classic games for artificial intelligence owing to its enormous search space and the difficulty of evaluating board positions and moves