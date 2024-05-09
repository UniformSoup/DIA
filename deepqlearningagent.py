from tensorflow import keras
from keras import layers, Model, losses, saving, regularizers
from collections import deque
import numpy as np

class DeepQLearningAgent():
    def __init__(self, file="deepqlearning.keras", isLearning=False):
        self.file = file 
        self.isLearning = isLearning       
        #self.intervals = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
        self.actions = [-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4]
        self.learningRate = 0.01 # Alpha
        self.discountFactor = 0.99 # Gamma
        self.explorationRate = 0.3 # Epsilon
        self.decayRate = 0.995
        self.updateRate = 0.05
        self.min_batch_size = 64
        self.memory = deque(maxlen=1000)

        try:
            self.model = saving.load_model(file)
            self.model.compile(keras.optimizers.Adam(learning_rate=self.learningRate), losses.MeanSquaredError())
            self.target_model = keras.models.clone_model(self.model)
        except:
            self.model = self.build_model()
            self.target_model = keras.models.clone_model(self.model)

    def build_model(self):
        inputs = layers.Input(shape=(19,))
        model  = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
        model  = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(model)
        model  = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(model)
        model  = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(model)
        output = layers.Dense(len(self.actions), activation='linear')(model)
        model = Model(inputs, outputs=output)

        model.compile(keras.optimizers.Adam(learning_rate=self.learningRate), losses.Huber())
        model.summary()

        return model

    def getState(self, track):
        return track[np.newaxis]

    def learn(self, ob, action, reward, new_ob, done):
        self.memory.append((self.getState(ob[6]), np.where(self.actions == action[0])[0], reward, self.getState(new_ob[6]), done))

    def action(self, observation):
        current_state = self.getState(observation[6])
        throttle = np.clip(np.average(observation[6][9]) * 5, 0.0, 1.0)
            
        if self.isLearning and np.random.rand() < self.explorationRate:
            direction = np.random.choice(self.actions)
            print("Random: ", direction)
            return (direction, throttle)
        else:
            direction = self.actions[np.argmax(self.model.predict_on_batch(current_state))]
            print("DQN: ", direction)
            return(direction, throttle)
    
    def save(self):
        if len(self.memory) < self.min_batch_size:
            return
        
        np.random.shuffle(self.memory)

        states, actions, rewards, new_states, dones = zip(*self.memory)

        qvalues = self.model.predict_on_batch(np.vstack(states))
        futureQvalues = self.target_model.predict_on_batch(np.vstack(new_states))

        for i in range(len(self.memory)):
            qvalues[i, actions[i]] = rewards[i] + self.discountFactor * np.max(futureQvalues[i]) * (~dones[i])

        self.memory.clear()

        self.model.fit(np.vstack(states), qvalues, verbose=False)
        self.explorationRate *= self.decayRate
        print(self.explorationRate)
        self.target_model = self.update_target_network(self.model, self.target_model)
        self.model.save(self.file)

    def update_target_network(self, model, target_model):
        model_weights = model.get_weights()
        target_weights = target_model.get_weights()

        for i in range(len(model_weights)):
            target_weights[i] = self.updateRate * (model_weights[i] - target_weights[i]) + target_weights[i]
            
        target_model.set_weights(target_weights)
        return target_model