import random, math
from parameters import *
from utils import *

class CliffWalk:
    
    def __init__(self):
        
        self.config = {
            "algorithms": ALGORITHMS,
            "on_policy": ON_POLICY,
            "epsilon": EPSILON,
            "decay_rate": DECAY_RATE,
            "learning_rate": LEARNING_RATE,
            "discount_factor": DISCOUNT_FACTOR,
            "step_size": STEP_SIZE,
            "episodes": EPISODES
        }
        
        self.history = {}
        self.reward_sums = {}
        self.time_taken = {}
        self.init_results()
        
        self.Q = {}
        self.init_Q()
        
        self.state = (0, 0)
        self.time = 0
        
        self.decayed_epsilon = self.config["epsilon"]
        
        self.computed = False
    
    def compute(self):
        
        algorithms, step_size, learning_rate, discount_factor, episodes = (
                self.config["algorithms"],
                self.config["step_size"],
                self.config["learning_rate"],
                self.config["discount_factor"],
                self.config["episodes"]
            )
        
        for algorithm in algorithms:
            self.init_Q()
            self.time = 0
            self.decayed_epsilon = self.config["epsilon"]
            
            for episode in range(episodes):
                
                print("computing episode " + str(episode))
                
                self.state = (0, 0)
                episode_time = 0
                episode_history = []
                
                if algorithm in TD_ALGORITHMS:
                    
                    terminating_time = math.inf
                    t = episode_time - step_size # t is the time of the state whose estimate we are updating
                    while t < terminating_time - 1:
                        
                        if episode_time < terminating_time:
                            # transition is a tuple of the form
                            # (S_t, A_t, R_t+1, S_t+1, A_t+1)
                            # if S_t+1 is terminal:
                            # (S_t, A_t, R_t+1, S_t+1)
                            transition = None
                            if episode_time == 0:
                                transition = self.generate_transition(self.state)
                            else:
                                # while generating transition one time step ago
                                # the action taken has been already determined in the ending 'SA' of the 'SARSA' tuple.
                                # so the starting 'SA' of this 'SARSA' transition is already computed.
                                # we need to use that action to not detach from the sequence of what is
                                # being experienced.
                                action_taken = episode_history[episode_time - 1][4]
                                transition = self.generate_transition(self.state, action_taken)
                                
                            episode_history.append(transition)
                            
                            self.state = transition[3]
                            if CliffWalk.state_terminal(self.state):
                                terminating_time = episode_time + 1
                        
                        episode_time += 1
                        self.increment_time()
                        
                        t = episode_time - step_size
                        
                        if t >= 0:
                            up_to = min(t + step_size, terminating_time)
                            
                            target = 0
                            
                            discount_factor_temp = 1 
                            for i in range(t + 1, up_to + 1):
                                target += ( discount_factor_temp ) * episode_history[i - 1][2]
                                discount_factor_temp = discount_factor_temp * discount_factor
                            
                            if algorithm == Algorithms.SARSA:
                                if t + step_size < terminating_time:
                                    _ts = episode_history[t + step_size - 1]
                                    target += ( discount_factor ** step_size ) * self.Q[ _ts[3] ][ _ts[4] ]
                            
                            if algorithm == Algorithms.Q_LEARNING:
                                if t + step_size < terminating_time:
                                    _ts = episode_history[t + step_size - 1]
                                    greedy_action = self.find_greedy_action( _ts[3] )
                                    target += ( discount_factor ** step_size ) * self.Q[ _ts[3] ][ greedy_action ]
                            
                            if algorithm == Algorithms.EXPECTED_SARSA:
                                if t + step_size < terminating_time:
                                    _ts = episode_history[t + step_size - 1]
                                    p = self.get_action_probabilities( _ts[3] )
                                    
                                    w_action_sum = 0
                                    for action in Actions:
                                        w_action_sum += p[action] * self.Q[ _ts[3] ][action]
                                    
                                    target += ( discount_factor ** step_size ) * w_action_sum
                            
                            s_u, a_u = ( episode_history[t][0], episode_history[t][1] )
                            self.Q[ s_u ][ a_u ] = self.Q[ s_u ][ a_u ] + ( learning_rate * ( target - self.Q[ s_u ][ a_u ] ) )
                
                if algorithm == Algorithms.MONTE_CARLO:
                    while not CliffWalk.state_terminal(self.state):
                        transition = None
                        if episode_time == 0:
                            transition = self.generate_transition(self.state)
                        else:
                            action_taken = episode_history[episode_time - 1][4]
                            transition = self.generate_transition(self.state, action_taken)
                        
                        self.state = transition[3]
                        episode_history.append(transition)
                        episode_time += 1
                        self.increment_time()
                    
                    for i in range( len( episode_history ) ):
                        
                        sa_target_value = 0
                        discount_factor_temp = 1
                        for j in range(i, len(episode_history)):
                            sa_target_value += ( discount_factor_temp ) * episode_history[j][2]
                            discount_factor_temp = discount_factor_temp * discount_factor
                        
                        s_u, a_u = ( episode_history[i][0], episode_history[i][1] )
                        self.Q[ s_u ][ a_u ] = self.Q[ s_u ][ a_u ] + ( learning_rate * ( sa_target_value - self.Q[ s_u ][ a_u ] ) )
                    
                self.history[algorithm].append(episode_history)
                
                reward_sum = 0
                for transition in episode_history:
                    reward_sum += transition[2]
                
                self.reward_sums[algorithm].append(reward_sum)
        
        self.computed = True
        return

    def find_greedy_action(self, state):
        
        tie_list = []
        
        max_value = -math.inf
        for action, value in self.Q[state].items():
            if value > max_value:
                max_value = value
                tie_list = [ action ]
            elif value == max_value:
                tie_list.append(action)
        
        return random.choice( tie_list )
    
    def pick_action(self, state):
        on_policy, epsilon, decay_rate = ( self.config["on_policy"], self.config["epsilon"], self.config["decay_rate"] )
        
        action = None
        
        greedy_action = self.find_greedy_action(state)
        
        if on_policy == Policies.GREEDY:
            action = greedy_action
        
        if on_policy == Policies.EPSILON_GREEDY:
            if random.random() < epsilon:
                action = random.choice( list(Actions) )
            else:
                action = greedy_action
        
        if on_policy == Policies.DECAYING_EPSILON_GREEDY:
            if random.random() < self.decayed_epsilon:
                action = random.choice( list(Actions) )
            else:
                action = greedy_action
        
        return action
    
    def increment_time(self):
        self.time += 1
        self.decayed_epsilon = self.decayed_epsilon * self.config["decay_rate"]
    
    def get_action_probabilities(self, state):
        
        on_policy, epsilon = ( self.config["on_policy"], self.config["epsilon"])
        
        action_probabilities = {
            Actions.UP: 0,
            Actions.DOWN: 0,
            Actions.RIGHT: 0,
            Actions.LEFT: 0
        }
        
        greedy_tie_list = []
        
        max_value = -math.inf
        for action, value in self.Q[state].items():
            if value > max_value:
                max_value = value
                greedy_tie_list = [ action ]
            elif value == max_value:
                greedy_tie_list.append(action)
        
        
        if on_policy == Policies.GREEDY:
            weight = 1 / len(greedy_tie_list)
            
            for action in greedy_tie_list:
                action_probabilities[action] = weight
        
        if on_policy == Policies.EPSILON_GREEDY:
            weight = (1 - epsilon) / len(greedy_tie_list)
            for action in greedy_tie_list:
                action_probabilities[action] = weight
            
            for action in action_probabilities:
                action_probabilities[action] += ( epsilon / 4 )
        
        if on_policy == Policies.DECAYING_EPSILON_GREEDY:
            weight = (1 - self.decayed_epsilon) / len(greedy_tie_list)
            for action in greedy_tie_list:
                action_probabilities[action] = weight
            
            for action in action_probabilities:
                action_probabilities[action] += ( self.decayed_epsilon / 4 )

        return action_probabilities
                    

    def generate_transition(self, state, action=None):

        if action == None:
            action = self.pick_action(state)
        
        next_state = self.next_state(state, action)
        
        transition = None
        if CliffWalk.state_terminal(next_state):
            transition = (
                state,
                action,
                CliffWalk.get_reward(state, action),
                next_state
            )
        else:
            transition = (
                state,
                action,
                CliffWalk.get_reward(state, action),
                next_state,
                self.pick_action(next_state)
            )
        
        return transition
    
    def change_config(self, property, new_value):
        self.config[property] = new_value
        self.reset()
    
    def init_Q(self):        
        for x in range(0,12):
            for y in range(0, 4):
                state = (x, y)
                
                self.Q[state] = {
                    Actions.UP: 0,
                    Actions.DOWN: 0,
                    Actions.LEFT: 0,
                    Actions.RIGHT: 0
                }
        
    def init_results(self):
        self.history = {}
        self.reward_sums = {}
        self.time_taken = {}
        
        for algorithm in self.config["algorithms"]:
            self.history[algorithm] = []
            self.reward_sums[algorithm] = []
            self.time_taken[algorithm] = 0
    
    def reset(self):
        self.init_results()
        self.init_Q()
        self.decayed_epsilon = self.config["epsilon"]
        
        self.state = (0, 0)
        self.time = 0
        
        self.computed = False
    
    def get_history(self):
        return self.history

    def get_reward_sums(self):
        return self.reward_sums
    
    # this can easily be refactored to return a ( state, reward ) tuple
    # corresponding more closely to formal p(s_t+1, r_t+1 | s_t, a_t)
    # that's unnecessary complexity in this example therefore avoided
    @staticmethod
    def next_state(state, action):
        x, y = state
        
        if y == 0 and action == Actions.DOWN:
            return (x, y)
        
        if y == 3 and action == Actions.UP:
            return (x, y)
        
        if x == 0 and action == Actions.LEFT:
            return (x, y)
        
        if x == 11 and action == Actions.RIGHT:
            return (x, y)
        
        if action == Actions.DOWN:
            return (x, y - 1)
        
        if action == Actions.UP:
            return (x, y + 1)
        
        if action == Actions.LEFT:
            return (x - 1, y)
        
        if action == Actions.RIGHT:
            return (x + 1, y)
        
        raise ValueError("Action not provided")
    
    @staticmethod
    def state_terminal(state):
        x, y = state
        
        if ( x < 11 and x > 0 ) and y == 0:
            return True
        
        if x == 11 and y == 0:
            return True
        
        return False
    
    # rewards are formally associated with transitions i.e.
    # state-action pairs.
    @staticmethod
    def get_reward(state, action):
        x, y = CliffWalk.next_state(state, action)
        
        if ( x < 11 and x > 0 ) and y == 0:
            return -100
        
        return -1