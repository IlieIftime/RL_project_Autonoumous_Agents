import gymnasium as gym
import numpy as np
import random
import tabular_sarsa as Tabular_SARSA
import matplotlib.pyplot as plt
import sys
import time

class SARSA(Tabular_SARSA.Tabular_SARSA):
    def __init__(self):
        super(SARSA, self).__init__()

    # TODO
    def learn_policy(self, env, gamma, learning_rate, epsilon, lambda_value, num_episodes):
    # Hyperparameters (MODIFY THESE TO OPTIMIZE PERFORMANCE)
        
        self.gamma = gamma              # <----- [0.9-0.99] Discount factor (recommended: 0.99)
        self.alpha = learning_rate      # <----- [0.1-0.3] Learning rate (recommended: 0.2)
        self.epsilon = epsilon          # <----- [0.05-0.2] Exploration rate (recommended: 0.1)
        self.lambda_value = lambda_value # <----- [0.5-0.9] Eligibility trace decay (recommended: 0.7)
        
        # Secondary hyperparameters
        epsilon_decay = 0.995           # <----- [0.99-0.999] Epsilon decay rate
        min_epsilon = 0.01              # <----- Minimum exploration probability
        max_steps = 200                 # <----- Max steps per episode

        start_time = time.time()
        rewards_history = []
        steps_history = []

        # Initialize Q-table and eligibility traces
        if self.qtable is None:
            self.num_states = env.observation_space.n
            self.num_actions = env.action_space.n
            self.qtable = np.zeros((self.num_states, self.num_actions))
            self.policy = np.zeros((self.num_states, 1), dtype=int)
        
        for episode in range(num_episodes):
            # Reset eligibility traces
            self.etable = np.zeros((self.num_states, self.num_actions))
            
            state = env.reset()[0]
            action = self.LearningPolicy(state)
            done = False
            episodic_reward = 0
            steps = 0

            while not done and steps < max_steps:
                # Take action and observe next state
                next_state, reward, done, _, _ = env.step(action)
                next_action = self.LearningPolicy(next_state)
                
                # Calculate TD error
                td_error = reward + self.gamma * self.qtable[next_state, next_action] * (not done) - self.qtable[state, action]
                
                # Update eligibility trace
                self.etable[state, action] += 1
                
                # Update Q-values using eligibility traces
                self.qtable += self.alpha * td_error * self.etable
                
                # Decay eligibility traces
                self.etable *= self.gamma * self.lambda_value
                
                # Update policy
                self.policy[state, 0] = np.argmax(self.qtable[state, :])
                
                # Move to next state
                state, action = next_state, next_action
                episodic_reward += reward
                steps += 1

                # Timeout check
                if time.time() - start_time > 30:
                    print("Timeout! Stopping training...")
                    done = True

            # Store results and decay epsilon
            rewards_history.append(episodic_reward)
            steps_history.append(steps)
            self.epsilon = max(min_epsilon, self.epsilon * epsilon_decay)

            # Progress logging
            if episode % 100 == 0:
                avg_steps = np.mean(steps_history[-100:] if episode >= 100 else steps_history)
                print(f"Episode {episode}: Steps={steps}, Reward={episodic_reward}, ε={self.epsilon:.3f}")

        # Final evaluation
        total_time = time.time() - start_time
        avg_steps_last_100 = np.mean(steps_history[-100:]) if num_episodes >= 100 else np.mean(steps_history)
        
        print("\nFinal results:")
        print(f"- Total time: {total_time:.2f}s {'PASS' if total_time <= 30 else 'FAIL'}")
        print(f"- Average steps (last 100): {avg_steps_last_100:.1f} {'PASS' if avg_steps_last_100 <= 30 else 'FAIL'}")
        print(f"- Average reward: {np.mean(rewards_history[-100:]):.1f}")

        # Save files for evaluation
        np.save("qvalues_taxi_sarsa_lambda_avaliacao", self.qtable)
        np.save("policy_taxi_sarsa_lambda_avaliacao", self.policy)

        return self.policy, self.qtable, rewards_history

    def LearningPolicy(self, state, testing=False):
        return Tabular_SARSA.Tabular_SARSA.learningPolicy(self, state, testing=testing)

def plot_rewards(episode_rewards):
    """
    Plots a learning curve for SARSA

    Input:
        episode_rewards: a list of episode rewards

    """
    plt.plot(episode_rewards)
    plt.ylabel("rewards per episode")
    plt.ion()
    plt.savefig("rewards_plot_lambda.png")

def render_visualization(learned_policy):
    """
    Renders a taxi problem visualization

    Input:
        learned_policy: the learned SARSA policy to be used by the taxi

    """
    env = gym.make("Taxi-v4", render_mode="human")
    state = env.reset()[0]
    env.render()
    while True:
        next_state, reward, done, _, info = env.step(learned_policy[state, 0])
        env.render()
        print("Reward: {}".format(reward))
        state = next_state
        if done:
            break
    env.close()

def avg_episode_rewards(num_runs):
    """
    Runs the learner algorithms a number of times and averages the episodic rewards
    from all runs for each episode

    Input:
        num_runs: the number of times to run the SARSA learner

    Output:
        episode_rewards: a list of averaged rewards per episode over a num_runs number of times
        learned_policy: the policy learned by the last run of sarsaLearner.learn_policy() to be
        used in problem visualization
    """
    episode_rewards = []
    for _ in range(num_runs):
        env = gym.make("Taxi-v4")
        env.reset()
        sarsaLearner = SARSA()
        learned_policy, q_values, single_run_er = sarsaLearner.learn_policy(
            env, 0.95, 0.2, 0.1, 0.8, 1000
        )         # single_run_er is the episodic reward for each run

        if not episode_rewards: # on the first iteration of this loop, episodeRewards will be empty
            episode_rewards = single_run_er
        else: # add this run's ERs to previous runs in order to calculate the average later
            episode_rewards = [episode_rewards[i] + single_run_er[i] for i in range(len(single_run_er))]

    # Get the average over ten runs
    episode_rewards = [er / num_runs for er in episode_rewards]

    return episode_rewards, learned_policy

def test_policy():
    """
    Tests the learned policy on the Taxi-v4 environment.
    """
    policy = np.load("policy_taxi_sarsa_lambda_avaliacao.npy")
    env = gym.make("Taxi-v4", render_mode="human")
    env.reset()
    num_episodes = 10
    rewards_each_test_episode = []
    steps_each_test_episode = []

    for _ in range(num_episodes):
        state = env.reset()[0]
        episodic_reward = 0
        steps = 0
        while True:
            action = policy[state][0]
            next_state, reward, done, _, info = env.step(action)
            steps += 1

            episodic_reward += reward

            state = next_state

            if done or steps >= 200:
                break

        rewards_each_test_episode.append(episodic_reward)
        steps_each_test_episode.append(steps)

    print("Rewards each test episode: {}".format(rewards_each_test_episode))
    print("Steps each test episode: {}".format(steps_each_test_episode))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        "*** run without input arguments to learn ***"
        episode_rewards, learned_policy = avg_episode_rewards(10)
        plot_rewards(episode_rewards)
        render_visualization(learned_policy)
    elif sys.argv[1] == "test":
        "*** run to test the saved policy with input argument test ***"
        test_policy()
    else:
        print("unknown input argument")
