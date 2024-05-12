from qlearningagent import QLearningAgent
from deepqlearningagent import DeepQLearningAgent
from basicagent     import BasicAgent
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pickle
from gym_torcs import TorcsEnv

def results(filename, agent):
    try:
        with open(filename, 'rb') as file:
            stats = pickle.load(file)
        return stats
    except:
        done = False
        env = TorcsEnv(vision=False, throttle=True)
        distances = []
        laptimes = []
        ob = env.reset(relaunch=True)
        while len(laptimes) < 50:
            action = agent.action(ob)
            ob, _, done, _ = env.step(action)

            # get last laptime.
            if ob[8] != 0 and (len(laptimes) == 0 or ob[8] != laptimes[-1]):
                print(f'Laptime #{len(laptimes)}: ', ob[8])
                laptimes.append(ob[8])
                stats = {'laptimes' :  laptimes, 'distances' : distances}
                with open(filename, 'wb') as file:
                    pickle.dump(stats, file)

            # if crashed, record the distance raced before a crash.
            if done:
                print(f'Distance #{len(distances)}: ', ob[9])
                distances.append(ob[9])
                ob = env.reset(relaunch=True)
                # introduce a random start to the basic agent,
                # qlearning agent has 2% random moves.
                if isinstance(agent, BasicAgent):
                    agent.throttle = np.random.sample()
                    agent.direction = np.random.sample() - 0.5

        env.end()
        stats = {'laptimes' :  laptimes, 'distances' : distances}

        with open(filename, 'wb') as file:
            pickle.dump(stats, file)

        return stats

basic_stats = results('basic_stats.pkl', BasicAgent())
qlearning_stats = results('qlearning_stats.pkl', QLearningAgent(isLearning=True, epsilon=0.02))
qlearning_training_rewards = np.load('training_QLearning.npy')

print("Basic Agent Stats.")
print(f"Mean Laptime: {np.mean(basic_stats['laptimes'])}")
print(f"Standard Deviation: {np.std(basic_stats['laptimes'])}")
print(f"Mean Distance: {np.mean(basic_stats['distances'])}")
print(f"Standard Deviation: {np.std(basic_stats['distances'])}")
print(f"Crash Count: {len(basic_stats['distances'])}")

print("QLearning Agent Stats.")
print(f"Mean Laptime: {np.mean(qlearning_stats['laptimes'])}")
print(f"Deviation: {np.std(qlearning_stats['laptimes'])}")
print(f"Mean Distance: {np.mean(qlearning_stats['distances'])}")
print(f"Standard Deviation: {np.std(qlearning_stats['distances'])}")
print(f"Crash Count: {len(qlearning_stats['distances'])}")

# uncomment this to plot the model, if graphvis is installed
# https://graphviz.gitlab.io/download/
#model = keras.models.load_model('deepqlearning.keras')
#keras.utils.plot_model(model, to_file='model.eps', show_shapes=True)


# Plot Rewards
fig, ax = plt.subplots()
ax.plot(qlearning_training_rewards)
ax.set_xlabel('Episodes')
ax.set_ylabel('Mean Reward')

plt.tight_layout()
plt.savefig('rewards.eps', format='eps')
plt.show()

# Plot Laptimes
fig, ax = plt.subplots()

ax.violinplot([basic_stats['laptimes'], qlearning_stats['laptimes']], showmedians=True)

ax.set_xticks([1, 2])
ax.set_xticklabels(['Zero-Memory', 'Q-Learning'])
ax.set_ylabel('Laptime (seconds)')

plt.tight_layout()
plt.savefig('violin.pdf', format='pdf')
plt.show()

# Plot distances before crash.
fig, ax = plt.subplots()

ax.boxplot([np.array(basic_stats['distances']), qlearning_stats['distances']])

ax.set_xticks([1, 2])
ax.set_xticklabels(['Zero-Memory', 'Q-Learning'])
ax.set_ylabel('Distance Raced (meters)')

plt.tight_layout()
plt.savefig('boxplot.eps', format='eps')
plt.show()

# Plot crash count
fig, ax = plt.subplots()

ax.bar(0.2, len(basic_stats['distances']), width=0.3)
ax.bar(0.6, len(qlearning_stats['distances']), width=0.3)

ax.set_xticks([0.2, 0.6])
#ax.set_yticks([len(basic_stats['distances']), len(qlearning_stats['distances'])])
ax.set_xticklabels(['Zero-Memory', 'Q-Learning'])
ax.set_ylabel('Crash Counts')

plt.tight_layout()
plt.savefig('bar.eps', format='eps')
plt.show()