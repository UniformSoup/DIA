from agent import Agent
import numpy as np

class QLearningAgent(Agent):
    def __init__(self, file="qlearning.npy", isLearning=False, epsilon=0.2):
        self.file = file 
        self.isLearning = isLearning       
        #self.intervals = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
        self.distanceIntervals = np.linspace(0.0, 0.1, 10)[1:-1]
        self.actions = [-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4]
        self.learningRate = 0.15 # Alpha
        self.discountFactor = 0.5 # Gamma
        self.explorationRate = epsilon # Epsilon
        self.decayRate = 0.995

        try:
            self.qtable = np.load(file)
        except:
            self.qtable = np.zeros(
                (len(self.distanceIntervals) + 1, 
                 len(self.distanceIntervals) + 1, 
                 len(self.distanceIntervals) + 1,
                 len(self.actions)))
            np.save(file, self.qtable)

    def getState(self, track):
        left = np.digitize(np.min(track[:9]), self.distanceIntervals)
        middle = np.digitize(0.5 * track[9], self.distanceIntervals)
        right = np.digitize(np.min(track[10:]), self.distanceIntervals)

        return (left, middle, right)
    
    def updateTable(self, state, action, reward, next_state):
        oldQvalue = self.qtable[state][action]
        maxFutureQvalue = np.max(self.qtable[next_state])
        td = reward + self.discountFactor * maxFutureQvalue
        self.qtable[state][action] = self.learningRate * (td - oldQvalue) + oldQvalue

    def learn(self, ob, action, reward, new_ob, done):
        state = self.getState(ob[6])
        new_state = self.getState(new_ob[6])
        action_index = self.actions.index(action[0])
        self.updateTable(state, action_index, reward, new_state)
        print("State:", state)
        print("Reward:", reward)
        print("Exploration Rate:", self.explorationRate)

        if done:
            self.explorationRate *= self.decayRate

    def action(self, observation):
        current_state = self.getState(observation[6])
        choice = None
            
        if self.isLearning and np.random.rand() < self.explorationRate:
            choice = np.random.choice(len(self.actions))
        else:
            choice = np.argmax(self.qtable[current_state])

        throttle = np.clip(observation[6][9] * 5, 0.0, 1.0)

        return (self.actions[choice], throttle)
    
    def save(self):
        np.save(self.file, self.qtable)