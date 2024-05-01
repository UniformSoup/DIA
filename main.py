from gym_torcs import TorcsEnv
from basicagent import BasicAgent
from qlearningagent import QLearningAgent
from deepqlearningagent import DeepQLearningAgent
import numpy as np

episodes = 2000
steps = 1000
isTraining = True
patience = 20

print("""
Please press F2 when the game starts to see the drivers view.
Please pick an agent to run
1) Basic Agent (no memory, operates using only the current state)
2) QLearning Agent (5103 parameters)
3) DeepQLearning Agent (3657 parameters)
""")

choice = -1
agent = None

while True:
    try:
        choice = int(input())
    except:
        pass

    if choice > 0 and choice < 4:
        agent = {1 : BasicAgent(),
                 2 : QLearningAgent(isLearning=isTraining),
                 3 : DeepQLearningAgent(isLearning=isTraining)}[choice]
        break
    else:
        print("You have entered an invalid choice.")

env = TorcsEnv(vision=False, throttle=True, alt_reward=(isinstance(agent, QLearningAgent) and isTraining))

done = False
reward = 0
best_mean_reward = -np.inf
mean_rewards = []
patience_count = 0

for i in range(episodes):
    ob = env.reset(relaunch=(i % 5)) # relaunching prevents a memory leak in gym_torcs apparently.

    episode_rewards = []
    for j in range(steps):
        if j == steps - 1:
            done = True
    
        action = agent.action(ob, reward, done)
        ob, reward, done, _ = env.step(action)

        episode_rewards.append(reward)

        if done:
            print(f"Episode {i + 1} ended after {j+1} steps.")
            break

    mean_rewards.append(np.mean(episode_rewards))

    if mean_rewards[-1] > best_mean_reward:
        best_mean_reward = mean_rewards[-1]
        patience_count = 0
    else:
        patience_count += 1

        if patience_count > patience:
            break

agent_name = {1 : "Basic", 2 : "QLearning", 3 : "DeepQLearning"}[choice]
np.save(f"training_{agent_name}.npy", mean_rewards)
env.end()
