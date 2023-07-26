# SOC23_Breakout_Genius
WnCC SOC Season of Code - Breakout Genius: Using RL to Build AI game master

Explanation of the final code which implemented:
1) make_env function will be used to create an environment for the Atari game. it uses AtariPreprocessing, FrameStack, and TransformReward
2) DQNAgent class consists of Convolutional and linear layers and includes methods for obtaining Q-values, sample actions, etc.
3) ReplayBuffer Class is used to store and manage replay buffers, required for training the DQN.
4) Main training loop starts the training process of DQN. it plays the game and does all those steps related to back-propagation, etc.
5) Agent uses epsilon-greedy exploration strategy, starting with probability start_epsilon and ending with end_epsilon
6) Evaluating using greedy action selection and recording mean rewards across episodes
7) Model is saved at regular intervals


Progress report until now: https://docs.google.com/document/d/15kdWirwxBqIiY0t8DJgzKWBiwpKJjsMhCBjm4DrRcr4/edit?usp=sharing

Notes on Reinforcement Learning: https://drive.google.com/file/d/1_yy4iHW69M0xtL2rAQTQDY2EtG_P88sb/view?usp=sharing

Resources shared by my Mentors: https://docs.google.com/document/d/1Iwq53YUAL_azGlDw87LgkWMROnrN5jGh0vooQ4XryoA/edit?usp=sharing
