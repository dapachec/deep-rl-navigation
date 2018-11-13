# Project Details

In this project, we will train a virtual agent to solve the Banana Collection task, a benchmark in the deep-RL comunity developed by the Unity ML-agents team. 

The objective of our navigating agent is to collect as many as possible yellow bananas while avoiding to collect blue bananas. 

The observations from the environment are stored in a continuous space of 37 dimensions, including the agent's velocity, along with a ray-based perception of objects around the agent's forward direction. In particular, the agent throws 7 rays at angles (20, 90, 160, 45, 135, 70, 100). Each ray contains info about the presence of one of four detectable objects encoded as a one-shot vector (i.e., yellow banana, wall, blue banana, other agent). It also contains a distance measure with respect to the collided object (a fraction of the ray's total lenght)

There are four possible actions available to the agent at each time step (Move forward, Move backward, Turn left, Turn right). 
The agent receives a reward of +1 when it collects a yellow banana, and of -1 for a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

In this simplified version, the environment is considered solved when the agent collects an average of > 13 yellow bananas over 100 consecutive episodes. 


# Getting Started

The project requires Python 3.6 or higher with the following libraries 

unityagents 
numpy 
PyTorch

You will also need the Unity Banana Collector environment, which can be downloaded [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip).


# Installation
Pre-requisites
Make sure you having a working version of Anaconda on your system.

First > clone the repo:
```
git clone 
```

Second > Install Dependencies
Create an anaconda environment that contains all the required dependencies to run the project.

```
conda create --name drlnd_navigation python=3.6
source activate drlnd_navigation
conda install -y python.app
conda install -y pytorch -c pytorch
pip install unityagents
```

# Instructions

To run the code in the repository, please run main.py from the terminal. 
This file will call the dqn.train function in the main_functions script. 

To observe the behavior of the trained agent, please call the dqn.test function specifying the number of episodes you would like to observe.







