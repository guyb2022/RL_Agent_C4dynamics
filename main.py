from C4dynamicsEnv import C4dynamicsEnv
from C4dAgent import C4dAgent
import numpy as np
import pandas as pd
import os.path
import matplotlib as plt
import time
import warnings
import argparse

warnings.filterwarnings("ignore", category=UserWarning)


def main(args):
    # Parse command-line argument
    start_time = time.time()
    agent_name = args.agent_name
    data_file = args.data_file
    batch_size = args.batch_size
    n_episodes = args.n_episodes

    print('----------------------------------------------------')
    print("---------------  Starting the program --------------")
    print('----------------------------------------------------')
    print("Loading data........................................")
    # Load the simulation data
    data = pd.read_csv(data_file)

    # Create the environment
    print("Creating the Env....................................")
    env = C4dynamicsEnv(data, agent_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(f"State size: {state_size} Action size: {action_size}")

    # Create the agent
    print("Creating agent......................................")
    agent = C4dAgent(state_size, action_size, agent_name)
    n_episodes = 1000
    batch_size = 32
    best_distance = 1000000 # initialize to the max distance to target
    scores = []
    total_list_distance = []

    print("Starting Main loop..................................")
    print("----------------------------------------------------")
    for episode in range(1, n_episodes+1):
        # Initialize the environment and get the initial state
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        # Initialize the score for the episode
        score = 0
        while True:
            # Choose an action and act on it in the environment
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # Save the experience in the replay buffer
            agent.remember(state, action, reward, next_state, done)
            # Update the state and score
            state = next_state
            score += reward
            # If the episode is done, break the loop
            if done:
                break
        # Train the agent on a batch of experiences from the replay buffer
        agent.replay(batch_size)
        scores.append(score)

        if env.best_distance > best_distance:
              best_distance = env.best_distance
              agent.save_model_weights()

        print(f"Episode {episode}/{n_episodes} --- Score: {round(score,2)} --- best_distance: {round(best_distance,10)} ")
        total_list_distance.append(env.best_distance)
    agent.save()
    if not os.path.exists(f'weights/{agent_name}_weights.h5'):
        agent.save_model_weights()


    print('----------------------------------------------------')
    print("-----------------  program Finished ----------------")
    print('----------------------------------------------------')
    print(f"   average score: {np.mean(total_list_distance)}")
    print(f'   STD: {np.std(total_list_distance)}')
    print(f'   Max: {np.max(total_list_distance)}')
    print(f'   Min: {np.min(total_list_distance)}')
    print("----------------------------------------------------")
    end_time= time.time()
    print(f"Program run for {round(end_time - start_time,3)} seconds")
    # Plot the scores
    plt.plot(total_list_distance)
    plt.title("Agent performance")
    plt.xlabel('Episode')
    plt.ylabel('Total Balance')
    plt.show()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="simulation.csv", help="simulation data file")
    parser.add_argument("--agent_model_file", type=str, default="models/agent_model.h5", help="Agent model file")
    parser.add_argument("--reward_model_file", type=str, default="models/reward_model.h5", help="Reward model file")
    parser.add_argument("--epsilon", type=float, default=1.0, help="epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="epsilon decay")
    parser.add_argument("--epsilon_min", type=float, default=0.01, help="epsilon minimum")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--discount_factor", type=float, default=0.99, help="discount factor")
    parser.add_argument("--n_episodes", type=int, default=200, help="number of episodes")
    parser.add_argument("--window_size", type=int, default=10, help="window size")
    parser.add_argument("--agent_name", type=str, default=1, help="Agent Name")
    args = parser.parse_args()
    main(args)

    """
    usage:
    python main.py --data_file=data/simulation.csv --batch_size=32 --agent_name=si,ulation_num_1 --n_episodes=1000
    """