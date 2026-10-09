import gymnasium as gym
import numpy as np
import tabular_sarsa as Tabular_SARSA
import matplotlib.pyplot as plt
import sys
import time



class SARSA(Tabular_SARSA.Tabular_SARSA):
    def __init__(self):
        super(SARSA, self).__init__()

    # TODO
    def learn_policy(self, env, gamma, learning_rate, epsilon, lambda_value, num_episodes):
    # Hiperparâmetros principais (MODIFICAR ESTES PARA OTIMIZAR)
        self.gamma = gamma              # <----- [0.9-0.99] Controla peso das recompensas futuras (recomendado: 0.95)
        self.alpha = learning_rate      # <----- [0.1-0.5] Taxa de aprendizagem (recomendado: 0.2)
        self.epsilon = epsilon          # <----- [0.05-0.3] Taxa de exploração inicial (recomendado: 0.1)
        
        # Hiperparâmetros secundários (ajustes finos)
        epsilon_decay = 0.995           # <----- [0.99-0.999] Taxa de decaimento do ε (recomendado: 0.995)
        min_epsilon = 0.01              # <----- [0.001-0.05] ε mínimo (recomendado: 0.01)
        max_steps_per_episode = 200     # <----- Limite de passos por episódio (fixo)

        start_time = time.time()
        rewards_history = []
        steps_history = []

        # Inicialização da Q-table (não modificar)
        if self.qtable is None:
            state_space_size = env.observation_space.n
            action_space_size = env.action_space.n
            self.qtable = np.zeros((state_space_size, action_space_size))
            self.policy = np.zeros((state_space_size, 1), dtype=int)

        for episode in range(num_episodes):
            state = env.reset()[0]
            action = self.LearningPolicy(state)
            done = False
            episodic_reward = 0
            steps = 0

            while not done and steps < max_steps_per_episode:
                # SARSA update (núcleo do algoritmo)
                next_state, reward, done, _, _ = env.step(action)
                next_action = self.LearningPolicy(next_state)
                
                td_error = reward + self.gamma * self.qtable[next_state, next_action] * (not done) - self.qtable[state, action]
                self.qtable[state, action] += self.alpha * td_error
                
                self.policy[state, 0] = np.argmax(self.qtable[state, :])
                state, action = next_state, next_action
                episodic_reward += reward
                steps += 1

                # Timeout de 30 segundos (critério de parada)
                if time.time() - start_time > 30:
                    print("⏱️ Tempo excedido! Parando treino...")
                    done = True

            # Armazenamento e decaimento
            rewards_history.append(episodic_reward)
            steps_history.append(steps)
            self.epsilon = max(min_epsilon, self.epsilon * epsilon_decay)

            # Log de progresso
            if episode % 100 == 0:
                avg_steps = np.mean(steps_history[-100:] if episode >= 100 else steps_history)
                print(f"Episódio {episode}: Passos={steps}, Recompensa={episodic_reward}, ε={self.epsilon:.3f}")

        # Resultados finais (CRÍTICO PARA OS TESTES)
        total_time = time.time() - start_time
        avg_steps_last_100 = np.mean(steps_history[-100:]) if num_episodes >= 100 else np.mean(steps_history)
        
        print(f"\n🔍 Resultados finais:")
        print(f"- Tempo total: {total_time:.2f}s {'✅' if total_time <= 30 else '❌'}")
        print(f"- Média de passos (últimos 100 episódios): {avg_steps_last_100:.1f} {'✅' if avg_steps_last_100 <= 30 else '❌'}")
        print(f"- Recompensa média: {np.mean(rewards_history[-100:]):.1f}")

        # Salvamento obrigatório (para avaliação)
        np.save("qvalues_taxi_sarsa_avaliacao", self.qtable)
        np.save("policy_taxi_sarsa_avaliacao", self.policy)

        return self.policy, self.qtable, rewards_history

    def LearningPolicy(self, state, testing=False):
        return Tabular_SARSA.Tabular_SARSA.learningPolicy(self, state, testing=testing)
    
def decode_taxi_state(state):
    """
    Decodes the taxi-v4 state into its components.
    """
    taxi_row = (state // (5 * 5 * 4)) % 5
    taxi_col = (state // (5 * 4)) % 5
    passenger_location = (state // 4) % 5
    destination = state % 4
    
    return taxi_row, taxi_col, passenger_location, destination

def plot_rewards(episode_rewards):
    """
    Plots a learning curve for SARSA

    Input:
        episode_rewards: a list of episode rewards

    """
    plt.plot(episode_rewards)
    plt.ylabel("rewards per episode")
    plt.ion()
    plt.savefig("rewards_plot_taxi_sarsa.png")

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
        env.render() # just update the data
        print("Reward: {}".format(reward))
        time.sleep(1)
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
            env, 0.95, 0.2, 0.1, 0.1, 1000
        )         # single_run_er is the episodic reward for each run

        if not episode_rewards: # on the first iteration of this loop, episodeRewards will be empty
            episode_rewards = single_run_er
        else: # add this run's ERs to previous runs in order to calculate the average later
            episode_rewards = [episode_rewards[i] + single_run_er[i] for i in range(len(single_run_er))]

    # Get the average over ten runs
    episode_rewards = [er / num_runs for er in episode_rewards]

    return episode_rewards, learned_policy

def test_policy():
    policy = np.load("policy_taxi_sarsa_avaliacao.npy")
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
