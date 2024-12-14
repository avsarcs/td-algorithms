from CliffWalk import CliffWalk
from parameters import *
from utils import *
import matplotlib.pyplot as plt
import turtle

walk_instance = CliffWalk()

walk_instance.compute()

def get_episode_reward_average(reward_sums):
    i = 1
    total_sum = 0
    for reward_sum in reward_sums:
        total_sum += reward_sum
        i += 1

    return total_sum / i

# returns heading
def get_turtle_action(sarsa):
    if sarsa[0] == sarsa[3]:
        return None
    
    if sarsa[1] == Actions.RIGHT:
        return 0
    
    if sarsa[1] == Actions.UP:
        return 90
    
    if sarsa[1] == Actions.LEFT:
        return 180
    
    if sarsa[1] == Actions.DOWN:
        return 270
    
    return None    

choice = None
while choice != str(3):
    print("1.. sum of rewards graph")
    print("2.. see episode trace")
    print("3.. quit")
    choice = input()
    
    if choice == "1":
        
        for algo, reward_sums in walk_instance.reward_sums.items():
            print( algo_string_mapping[algo] + ": " + str(get_episode_reward_average(reward_sums)) )
            plt.plot(reward_sums, label=algo_string_mapping[algo])
        
        plt.title("Sum of Rewards. vs Episode for Different Algorithms")
        plt.xlabel("Episode")
        plt.ylabel("Sum of Rewards")
        plt.legend()
        
        plt.subplots_adjust(bottom=0.35)
        plt.figtext(
            0.5, 0.01,
            "On Policy: " + policy_string_mapping[ON_POLICY] + "\n"
            + " | Epsilon: " + str(EPSILON) + " | Decay Rate: " + str(DECAY_RATE) + "\n"
            + " | Learning Rate (alpha): " + str(LEARNING_RATE) + "\n"
            + " | Discount Factor (gamma): " + str(DISCOUNT_FACTOR) + "\n"
            + " | Step Size: " + str(STEP_SIZE),
            ha="center",
            fontsize=12
        )
        
        plt.show()
    
    if choice == "2":
        print("Which algorithm?")
        
        index = 1
        algo_choices = {}
        for algo in ALGORITHMS:
            print(str(index) + ".. " + algo_string_mapping[algo])
            algo_choices[index] = algo
            index += 1
        
        choice = int(input())
        algorithm = algo_choices[choice]
        
        print("Which episode? (0-" + str(EPISODES - 1)+")")
        episode = int(input())
        
        print(walk_instance.history[algorithm][episode])
        
        screen = turtle.Screen()
        screen.bgpic("cliff_terrain.png")
        screen.setup(width=1200, height=400)
        
        algo_turtle = turtle.Turtle()
        algo_turtle.penup()
        algo_turtle.goto(-550, -150)
        algo_turtle.pendown()
        
        if len(walk_instance.history[algorithm][episode]) > 50:
            algo_turtle.speed(8)
        
        for sarsa in walk_instance.history[algorithm][episode]:
            heading = get_turtle_action(sarsa)
            if heading != None:
                algo_turtle.setheading(heading)
                algo_turtle.forward(100)
                
        turtle.done()