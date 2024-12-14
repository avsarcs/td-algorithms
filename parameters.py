from utils import Algorithms, Policies

ALGORITHMS=[Algorithms.SARSA, Algorithms.Q_LEARNING]
ON_POLICY=Policies.EPSILON_GREEDY
EPSILON=0.1
DECAY_RATE = 0.95
LEARNING_RATE=0.1
DISCOUNT_FACTOR=0.2
STEP_SIZE=2
EPISODES=300