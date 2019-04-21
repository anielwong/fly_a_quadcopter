# Train a Quadcopter to fly
In this project, an agent that can fly a quadcopter using a reinforcement learning algorithm is trained. 

## Project Overview
The goal of this project is to train an quadcopter how to fly autonomously using reinforcement learning algorithms. Given a task, the agent learns to perform it without any help.

This project is based on the [Machine Learning Engineer Udacity Course](https://eu.udacity.com/course/machine-learning-engineer-nanodegree--nd009).

## Project Highlights
This project is designed to give a hands-on experience with Reinforcement Learning (RL) and how to build an agent that can fly a quadcopter. 

Subjects learned by completing the project:

	Dynamic Programming
	Monte Carlo Methods
	Temporal-Difference Methods
	RL in continuous space
	Deep Q-Learning
	Policy Gradients
	Actor-Critic Methods

This project contains several files:

    Quadcopter_Proeject.ipynb: This is the main file where the project is performed.
    physics_sim.py : This file contains the simulator for the quadcopter. DO NOT MODIFY THIS FILE.
    agents : This folder contain the built agents that flies the quadcopter.
    task.py : This file contains the task (environment) to perform.

In this project, `task.py` will define the task chosen. A first draft of task (policy_search.py) is provided to start with.

`agent.py` also design a reinforcement learning (RL) agent to complete the chosen task

## Project Instructions

### Instructions

1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/anielwong/fly_a_quadcopter.git
cd fly_a_quadcopter
```

2. Open the notebook.
```
jupyter notebook Quadcopter_Project.ipynb
```

3. Follow the instructions and tweak hyperparameters at will. 

