import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm

NUM_PROBLEMS = 2000
NUM_ARMS = 25
NUM_STEPS = 1000

EPSILONS = [0., 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
STABILITY_EPS = 1e-8

def explore_or_exploit(n_items, EPSILON):
    probs = np.random.random(n_items)
    explore = probs < EPSILON
    return explore


def main():

    # creating the q_*(a) for all a and all problems at the same time
    optimal_values = np.random.randn(NUM_PROBLEMS, NUM_ARMS)

    for EPSILON in EPSILONS:

        print(f"Evaluating epsilon-greedy with epsilon = {EPSILON}")

        # table containing found reward values for each arm and problem
        total_rewards = np.zeros((NUM_PROBLEMS, NUM_ARMS))
        total_actions = np.zeros((NUM_PROBLEMS, NUM_ARMS))
        expected_values = np.zeros((NUM_PROBLEMS, NUM_ARMS))

        # average reward_per_time_step
        avg_reward_tracker = []

        for i in tqdm(range(NUM_STEPS)):

            # are we exploring or exploiting for this specific problem?
            explore = explore_or_exploit(NUM_PROBLEMS, EPSILON)

            # define greedy move
            greedy = np.argmax(expected_values, axis=1)
            # define random move
            random = np.random.randint(0, 10, NUM_PROBLEMS)

            # select random when explore is true and greedy when explore is false
            actions = random * explore + greedy * np.logical_not(explore)

            rewards = optimal_values[np.arange(NUM_PROBLEMS), actions] + np.random.randn(NUM_PROBLEMS)

            total_rewards[np.arange(NUM_PROBLEMS), actions] += rewards
            total_actions[np.arange(NUM_PROBLEMS), actions] += 1
            
            expected_values = total_rewards / (total_actions + STABILITY_EPS)

            avg_reward_tracker.append(np.mean(rewards))

        plt.plot(np.arange(len(avg_reward_tracker)), avg_reward_tracker, label=f"eps={EPSILON}")

    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.savefig("avg_reward.png")



if __name__ == "__main__":
    main()