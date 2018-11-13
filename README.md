# Project Details

In this project, we will train a virtual agent to solve the Banana Collection task, a benchmark in the deep-RL comunity developed in the framework of the Unity ML-agents project. 

The objective of the agent in this task is to collect yellow bananas distributed in a rectangular virtual space, while avoiding to collect blue bananas.

The agent collects observations from the environment that are stored in a continuous space of 37 dimensions, including its velocity, and a ray-based perception of objects around its forward direction. Specifically, the agent throws 7 rays at angles (20, 90, 160, 45, 135, 70, 100), which contain info about the presence of one of four detectable objects encoded as a one-shot vector (i.e., yellow banana, wall, blue banana, other agent). Rays also contain a distance measure with respect to any collided object (a fraction of the ray's total lenght).

There are four possible actions available to the agent at each time step (Move forward, Move backward, Turn left, Turn right). 
The agent receives a reward of +1 when it collects a yellow banana, and of -1 for a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

In this simplified version, the environment is solved when the agent collects an average of > 13 yellow bananas over 100 consecutive episodes. 


# Getting Started

The project requires Python 3.6 or higher with the libraries unityagents, numpy, PyTorch.

You will also need the Unity Banana Collector environment, which can be downloaded [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip).


# Installation
Make sure you having a working version of Anaconda on your system.

1) Clone the repo:
```
git clone https://github.com/dpachec/deep-rl-navigation.git
```

2) Install Dependencies
Create an anaconda environment that contains all the required dependencies to run the project.

```
conda create --name drlnd_navigation python=3.6
source activate drlnd_navigation
conda install -y python.app
conda install -y pytorch -c pytorch
pip install unityagents
```

# Instructions

To train the agent run train.py. This script will train the agent until the task is solved, and will store the weigths of a neural network used to estimate the action value function when finished. 

To observe the behavior of the trained agent, run test.py, which will load the weights of the network from the saved file ("checkpoint_Nav_V05_17.pth"), and display a simulation with the agent behaving according to the learned policy. 







