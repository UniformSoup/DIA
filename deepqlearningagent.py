from agent import Agent
from tensorflow import keras
from keras import layers
import numpy as np
"""

class NeuralNetAgent(Agent):
    def __init__(self, file="model.keras", isLearning=False):
        self.file = file        
        self.isLearning = isLearning
        self.actions = [-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4]
        self.learningRate = 0.15 # Alpha
        self.discountFactor = 0.5 # Gamma
        self.explorationRate = 0.2 # Epsilon
        self.decayRate = 0.995 # Decay Rate
        self.state = None
        self.choice = None
        self.model = self.build_model()

        try:
            self.model.load_weights(file)
        except:
            pass

    def build_model(self):
        # All the sensors
        input = keras.Input(shape=(21,)) # track sensors (19) + last action (2)
        model_layers = layers.Dense(64, activation='sigmoid')(input)
        model_layers = layers.Dropout(0.2)(model_layers)
        model_layers = layers.Dense(32, activation='sigmoid')(model_layers)
        model_layers = layers.Dropout(0.1)(model_layers)
        output_direction = layers.Dense(1, activation='tanh', name='Direction')(model_layers)
        output_throttle = layers.Dense(1, activation='sigmoid', name='Throttle')(model_layers)
        
        model = keras.Model(inputs=input, 
                            outputs=[output_direction, output_throttle],
                            name="NeuralNetAgent")

        model.summary()

        return model.compile(
            optimizer="adam",
            loss='mse',
            metrics=['mse', 'mae'],
            steps_per_execution=1
            )
    
    def action(self, observation, reward, done):

        current_state = self.getState(observation[6])

        if self.isLearning and self.state != None:
            self.updateTable(self.state, self.choice, reward, current_state)
            print("State:", self.state)
            print("Reward:", reward)
            print("Exploration Rate:", self.explorationRate)

        if self.isLearning and np.random.rand() < self.explorationRate:
            self.choice = np.random.choice(len(self.actions))
        else:
            self.choice = np.argmax(self.qtable[current_state])

        if done:
            if np.count_nonzero(self.state) < 3:
                self.updateTable(self.state, self.choice, -1000, current_state)
            np.save(self.file, self.qtable)
            self.explorationRate *= self.decayRate
            self.state = None
        else:
            self.state = current_state

        throttle = np.clip(np.average(observation[6][9]) * 5, 0.0, 1.0)

        return (self.actions[self.choice], throttle)
    """

class DeepQLearningAgent:
    def __init__(self, file="deepqlearning.keras", isLearning=False):
        self.file = file
        self.isLearning = isLearning
        #self.actions = [-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4]
        self.actions = np.linspace(-0.4, 0.4, 9)
        self.learningRate = 0.15 # Alpha
        self.discountFactor = 0.5 # Gamma
        self.explorationRate = 0.2 # Epsilon
        self.decayRate = 0.995
        self.state = None
        self.choice = None
        self.model = None

        try:
            self.model.load(file)
        except:
            self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(19,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(len(self.actions), activation='linear')  # Output layer for Q-values
        ])
        model.summary()
        model.compile(optimizer='adam', loss='mse')
        return model

    def updateModel(self, state, action, reward, next_state):
        # Get old prediction
        target = self.model.predict(state)
        # Get the chosen action from the prediction
        oldQvalue = target[0][action]
        # Get the future best qvalue
        maxFutureQvalue = np.max(self.model.predict(next_state))
        # Calculate temporal difference
        td = reward + self.discountFactor * maxFutureQvalue
        # Update the old qvalue
        target[0][action] = self.learningRate * (td - oldQvalue) + oldQvalue
        # Apply this to the model
        self.model.fit(state, target, verbose=False)

    def action(self, observation, reward, done):
        # Turn [sensor1, ...] into [[sensor1], ...]
        current_state = observation[6][np.newaxis]

        # Error comparing numpy array to none, have to check if its an instance.
        if self.isLearning and isinstance(self.state, np.ndarray):
            self.updateModel(self.state, self.choice, reward, current_state)
            #print("Reward:", reward)
            print("Exploration Rate:", self.explorationRate)

        if self.isLearning and np.random.rand() < self.explorationRate:
            self.choice = np.random.choice(len(self.actions))
        else:
            self.choice = np.argmax(self.model.predict(current_state, verbose=False))

        if done:
            if np.min(observation[6]) < 0:
                self.updateModel(self.state, self.choice, -1, current_state)
            self.model.save(self.file)
            self.explorationRate *= self.decayRate
            self.state = None
        else:
            self.state = current_state

        throttle = np.clip(np.average(observation[6][9]) * 5, 0.0, 1.0)

        return (self.actions[self.choice], throttle)