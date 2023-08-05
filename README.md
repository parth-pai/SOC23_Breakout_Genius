# SOC23_Breakout_Genius
WnCC SoC Season of Code - Breakout Genius: Using RL to Build AI game master

Explanation of the Training code implemented:
1) make_env function will be used to create an environment for the Atari game. It uses AtariPreprocessing, FrameStack, and TransformReward.
2) DQNAgent class consists of Convolutional and linear layers and includes methods for obtaining Q-values, sample actions, etc.
3) ReplayBuffer Class is used to store and manage replay buffers, required for training the DQN.
4) Main training loop starts the training process of DQN. it plays the game and does all those steps related to back-propagation, etc.
5) Agent uses an epsilon-greedy exploration strategy, starting with probability start_epsilon and ending with end_epsilon.
6) Evaluating using greedy action selection and recording mean rewards across episodes.
7) Model is saved at regular intervals.

Explanation of Testing code implemented:
1) make_env function to create an environment for the Atari game using AtariPreprocessing, FrameStack, and TransformReward.
2) DQNAgent class consists of Convolutional and linear layers and includes methods for obtaining Q-values, sample actions, etc.
3) Generating animation and combining it to form a video
4) Loading the agent and obtaining the video output

Progress report: https://docs.google.com/document/d/15kdWirwxBqIiY0t8DJgzKWBiwpKJjsMhCBjm4DrRcr4/edit?usp=sharing

Resources shared by my Mentors: https://docs.google.com/document/d/1Iwq53YUAL_azGlDw87LgkWMROnrN5jGh0vooQ4XryoA/edit?usp=sharing

My Notes on Reinforcement Learning: https://drive.google.com/file/d/1_yy4iHW69M0xtL2rAQTQDY2EtG_P88sb/view?usp=sharing
