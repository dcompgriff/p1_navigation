# Project Report for 'p1_navigation'

Author: Daniel Griffin

## Introduction

The goal of this project was to use some form of reinforcement learning algorithm to create an agent that would learn over time to maximize it's score in the environment. The navigation environment consts of a unity environment with yellow and purple bananas. The learning agent recieves a reward of +1 when it picks up a yellow banana,  -1 when it picks up a blue banana, and 0 otherwise. The agent has 4 action options in this environment, and is passed a 37 dimension state object. Given this, the final goal is to create an agent that will achieve an average score of 13+ over a 100 windows of game play. (More information can be found in the project README.md file)

## General Approach

I opted to use the off-policy reinforcement learning method known as Q-Learning. Specifically, I opted to use a slightly modified version of the 'Deep Q Network' algorithm that I wrote for the DQN exercise. I modified the model portion of the code I wrote for the exercise so that it had 512 nodes in the first layer, and 128 nodes in the second layer. The algorithm followed the general methodology described in the 'Learning Algorithm' section. The algorithm took around 600 to 700 episodes to solve the envorinment, but I let the algorithm run until it reached a score of 15+ on average to provide some buffer room for the final evaluation of the average score over 100 runs in the environment.

## Learning Algorithm

### DQN At a High Level

The learning algorithm used was based off of the algorithm presented in the deep q network paper, which can be found [here](https://www.nature.com/articles/nature14236). The main algorithm follows the outline provided in the image provided below. The algorithm starts by initializing a 'replay memory' which is a fixed size memory buffer that holds previous experience tuples. The algorithm then initializes two neural networks. These are a 'target' network, and a 'current' network. The target network is used to generate the estimated value of the next state + action pair given the previous state and action. The target network allows learning while preventing target drift and unbounded oscillation during learning. The current network is used to perform updates. The network parameters are updated in a very similar method to computer memory circuits (See how D-Latches are used to prevent instability in D-Flip Flops [here](https://en.wikibooks.org/wiki/Electronics/Latches_and_Flip_Flops)). To prevent unbounded oscillation and instability, one network holds the current parameters while the other is then updated. When ready for the final update, the values from the current network are 'latched' into the target network's parameters. 

![DQN Learning Algorithm](https://github.com/dcompgriff/p1_navigation/blob/master/report_images/deep_q_network_algorithm.png)

### The steps of DQN

Once initialized, the Deep Q-Learning algorithm can be broken into 2 main concerns per time step, per episode. The first is sampling. During the sampling phase, the algorithm starts by choosing an action to take given the current state and policy. It then takes that action, and records the reward. It then adds that experience tuple to the replay memory (note here that I did not include the 'prepare next state' step as my algorithm does no extra preparation for this project). This continues for C steps. Once C steps have been taken, the algorithm moves on to the second concern, learning. 

Learning occurs by first obtaining a subsample of experience tuples from the experience replay buffer. Once retrieved, the data is converted into standard supervised learning pairs (xj, yj), where xj is the state representation of sj and yj is the estimated value of the best action (which uses the target network to estimage the recursive max parameter). Once the target is set, the data is fed to a neural network (see next section) which uses an adam optimizer to optimize the parameters. It's important to note that the network has an output for each action. This means that the task of estimating the best action for a state is framed as a multi-task learning problem. The main network is used for creating feature representations for all action value predictions (which makes this a multi-task regression problem). This means that on forward propagation, an estimated value for every action is produced. However, we want to only backpropagate the error for the actual action that was taken. So, we select this error, and zero the rest before performing backpropagation. Thus learning is achieved by using an experience buffer to generate supervised training batches, and the 'latching' mechanism is used to prevent unstable feeback loops due to the recursively defined neural network target value.

### My Network

The general network parameters is given below. The activation function used for each linear layer was a leaky Relu. 

![DQN Learning Algorithm](https://github.com/dcompgriff/p1_navigation/blob/master/report_images/network.png)

Other Parameters:
* Loss: MSE-Loss over a multi-task objective function
* Optimizer: Adam optimizer with default parameters and learning rate of .0005
* Network: Input vector with 37 elements into a FC linear network with bias and 512 nodes and a leaky Relu activation, 1 FC linear network with 128 nodes and a leaky Relu activation, and a final linear FC network with 4 actions.

### Differences from the paper

The main differences of my implementation from the paper's algorithm is that it is in general a simpler implementation. I used a non convolutional neural network model, a fixed vector of numbers (instead of images), and did not represent state sj as a sequence of 4 video frames. While I could have tried to add some of this extra serial dependence, I found that it was unnecessary as the agent solved the environment without the need for this added complexity. 

## Analysis of Results

![DQN Learning Algorithm](https://github.com/dcompgriff/p1_navigation/blob/master/report_images/training_results_13.png)

It took **560 episodes** to solve the environment. 

The above image is a plot of the rewards for this agent. I let the agent continue learning well past officially solving the environment by setting the solution threshold to '15.3' (Which is why the image says the environment was solved in 802 episodes. 15.3 was reached in 802 episodes). The above plot shows that for the first 400 episodes the agent steadily learns. At around 400 episodes we begin seeing the average performance start to taper off in its rate of growth.

![DQN Learning Algorithm](https://github.com/dcompgriff/p1_navigation/blob/master/report_images/training_results.png)

I then ran the algorithm for 100 episodes to calculate a final validation of the agent where no learning was performed (aka eps=0). The final validation score is given in the image below as an average score of 14.72 for 100 episodes (0-99). 

![DQN Learning Algorithm](https://github.com/dcompgriff/p1_navigation/blob/master/report_images/final_validation.png)

## Future Work

There are a few ways this work can be expanded. Obviously the algorithm can be expanded to incorporate the methods described in the Double Q Learning paper (Which generalizes the 'latching' mechanism), the Deuling Q networks Paper (which separates state estimation and state+action utilily estimation), and the Prioritized Replay Paper (Which uses weighted sampling of experiences to maximize learning on poorly performing data). The problem can also be extended by focusing more on the model itself. There has been a lot of work done with deep neural networks for image regcognition and object detection. We could attempt to first label samples of environment images to train a bannana location and estimation detector. We could then use this as input to the Q network model as a way to reason about yellow bananas, or even try to use this information in conjunction with a model building algorithm so that we can also incorporate planning based optimization in addition to the greedy sensory based actions. Other improvements might be to attempt to adapt some of the immitation learning work that has been going on for setting up a good baseline of behavior to learn from. We can have a human initially play the game to first gather experience tuples. We can then use this as an initial baseline of experience in the experience replay. Starting out, we can prioritize our sampling from this set of experiences over those initially generated from the environment. Over time, we can gradually loosen our prioritized sampling to be more like that of the prioritized experience replay algorithm. This may ultimately speed up the training of my agent. Some draw backs are that it makes heavier assumptions about the inintial experience being right, and may make it harder to find the most optimial policy as the neural network starts with higher biases about the right policy.

## Code Artifacts

* dqn_agent.py
    * Contains the DQN agent code that I implemented.
* model.py
    * Contains the pytorch based neural network model that I implemented for the project.
* Navigation.ipynb
    * Contains the code that runs the netowork and outputs the trained model weights.
* final_qnet_checkpiont.pth **saved model weights**
    * Contains the final weights for the model of the solved agent's neural network model.








