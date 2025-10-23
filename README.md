# Burning RED: Unlocking Subtask-Driven Reinforcement Learning and Risk-Awareness in Average-Reward Markov Decision Processes

This repository contains the official implementation for

**Burning RED: Unlocking Subtask-Driven Reinforcement Learning and Risk-Awareness in Average-Reward Markov Decision 
Processes**

*By: Juan Sebastian Rojas and Chi-Guhn Lee*

Reinforcement Learning Journal, 2025

**Abstract**: Average-reward Markov decision processes (MDPs) provide a foundational framework for sequential 
decision-making under uncertainty. However, average-reward MDPs have remained largely unexplored in reinforcement 
learning (RL) settings, with the majority of RL-based efforts having been allocated to discounted MDPs. In this work, 
we study a unique structural property of average-reward MDPs and utilize it to introduce Reward-Extended Differential 
(or RED) reinforcement learning: a novel RL framework that can be used to effectively and efficiently solve various 
learning objectives, or subtasks, simultaneously in the average-reward setting. We introduce a family of RED learning 
algorithms for prediction and control, including proven-convergent algorithms for the tabular case. We then showcase 
the power of these algorithms by demonstrating how they can be used to learn a policy that optimizes, for the first 
time, the well-known conditional value-at-risk (CVaR) risk measure in a fully-online manner, without the use of an 
explicit bi-level optimization scheme or an augmented state-space.

<p align="center">
  &#151; <a href="https://rlj.cs.umass.edu/2025/papers/Paper38.html"><b>View Paper</b></a> &#151;
</p>

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Results

To train the agents and reproduce the results in the paper, run the cells in the `red_pill_blue_pill.ipynb` and
`inverted_pendulum.ipynb` notebooks. A pair of html files with cell outputs that match our paper's results are provided 
for reference (`red_pill_blue_pill.html.zip` and `inverted_pendulum.html.zip`). 

## Citation 

```
@article{rojas2025burning,
    title={Burning {RED}: {U}nlocking Subtask-Driven Reinforcement Learning and Risk-Awareness in Average-Reward {M}arkov Decision Processes},
    author={Rojas, Juan Sebastian and Lee, Chi-Guhn},
    journal={Reinforcement Learning Journal},
    volume={6},
    pages={431--477},
    year={2025}
}
```

Please cite our paper if you use this code in your own work.
