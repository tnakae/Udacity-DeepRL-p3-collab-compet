[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


# About this project
**[Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)**
is MOOCs lecture by [Udacity](https://www.udacity.com/) in which DeepRL experts
teach basic knowledge and implementation techniques about Deep RL via online video
(mainly Youtube contents), and there are a lot of implementation tutorials
in which each student has to implement various DeepRL algorithms.

This repository of DeepRL source code is work of 3nd Project in this Nanodegree.
This is at the end of Lecture Part4, **Multi-Agent Reinforcement Learning**.
The project goal is to implement agent algorithm to play tennis in the simulator
using Unity ML-Agents([github repository](https://github.com/Unity-Technologies/ml-agents))

# Introduction
For this project, target simulator is [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

# Implemented algorithm
Udacity recommends to implement multi-agent-capable DDPG (MADDPG), but I used
policy based algorithm **PPO/GAE**. Please check out [Report.md](./Report.md)
about implementation details and performance reports.

# Getting Started
In this repository, there is best model already trained for enough steps,
but you can train in your environment following next procedures if you want.

1. Check [this nanodegree's prerequisite](https://github.com/udacity/deep-reinforcement-learning/#dependencies), and follow the instructions.

2. Clone this repository, change directory, and activate *drlnd* environment.
``` bash
git clone https://github.com/tnakae/Udacity-DeepRL-p3-collab-compet
chdir Udacity-DeepRL-p3-collab-compet
source activate drlnd
```

3. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

4. Place the file in the `bin/` folder, and unzip (or decompress) the file. 

5. Run the python scripts.
``` bash
# Train the agent
# Trrained model is saved to bestmodel.pth when achieved criteria
python train.py
# View the best model (bestmodel.pth) in the simulator
python view.py
# Plot the chart of best model scores (bestmodel_score.png)
python eval.py
```
