# Double-Deep-Q-Learning-for-Optimal-Execution
Double Deep Q-Learning for Optimal Execution


Optimal trade execution is an important problem faced by essentially all traders. Much research into optimal execution uses stringent model assumptions and applies continuous time stochastic control to solve them. Here, we instead take a model free approach and develop a variation of Deep Q-Learning to estimate the optimal actions of a trader. The model is a fully connected Neural Network trained using Experience Replay and Double DQN with input features given by the current state of the limit order book, other trading signals, and available execution actions, while the output is the Q-value function estimating the future rewards under an arbitrary action. We apply our model to nine different stocks and find that it outperforms the standard benchmark approach on most stocks using the measures of (i) mean and median out-performance, (ii) probability of out-performance, and (iii) gain-loss ratios.

________________________________________________________________________________________________________________________________________________________________________________


This Git repository is based on https://github.com/g0bel1n/DDQL-optimal-execution/tree/main, which is an implementation of Brian Ning, Franco Ho Ting Lin, Sebastian Jaimungal paper on Double Deep Q-learning (https://arxiv.org/abs/1812.06600), with some modifications added.
