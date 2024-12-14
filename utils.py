from enum import Enum

class Policies(Enum):
    GREEDY=1
    EPSILON_GREEDY=2
    DECAYING_EPSILON_GREEDY=3
    
class Algorithms(Enum):
    SARSA=1
    Q_LEARNING=2
    EXPECTED_SARSA=3
    MONTE_CARLO=4
    
class Actions(Enum):
    UP=1
    DOWN=2
    RIGHT=3
    LEFT=4

TD_ALGORITHMS = [Algorithms.SARSA, Algorithms.Q_LEARNING, Algorithms.EXPECTED_SARSA]

algo_string_mapping = {
            Algorithms.SARSA: "Sarsa",
            Algorithms.EXPECTED_SARSA: "Expected Sarsa",
            Algorithms.Q_LEARNING: "Q-Learning",
            Algorithms.MONTE_CARLO: "Monte Carlo"
        }
        
policy_string_mapping = {
    Policies.GREEDY: "Greedy",
    Policies.EPSILON_GREEDY: "Epsilon Greedy",
    Policies.DECAYING_EPSILON_GREEDY: "Decaying Epsilon Greedy"
}