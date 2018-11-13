# Project Details

The environment used in this project is the banana collection environment,  built in the Unity Game Engine. 

The objective of the agent is to collect as many as possible yellow bananas while avoiding to collect blue bananas. 

The observations that the agent extracts from the environment are stored in a continuous space of 37 dimensions, including the agent's velocity, along with a ray-based perception of objects around the agent's forward direction. In particular, the agent throws 7 rays at angles (20, 90, 160, 45, 135, 70, 100), with 90 being directly in front of the agent. Each ray contains info about the presence of one of four detectable objects encoded in a one-shot vector (i.e., yellow banana, wall, blue banana, other agent). There is also a distance measure which is a fraction of the ray length in the fifht position of each ray.

There are four possible actions available to the agent at each time step (Move forward, Move backward, Turn left, Turn right). 
The agent receives a reward of +1 when it collects a yellow banana, and of -1 for a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The environment is considered solved when the agent collects an average of > 13 yellow bananas over 100 consecutive episodes. 


# Getting Started

The project requires Python 3.6 or higher with the following libraries 

unityagents 
numpy 
PyTorch

You will also need the Unity Banana Collector environment, which can be downloaded here.

Instructions

To run the code in the repository, please run main.py from the terminal. Select the hiperparameters and set Train = True. 
To test the performance of the agent after learning, run the file with Train = False. 
