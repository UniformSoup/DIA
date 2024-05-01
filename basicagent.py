from agent import Agent
import numpy as np

class BasicAgent(Agent):
    def __init__(self):
        self.direction = 0
        self.throttle = 1.0
        self.threshold = 0.08

    def action(self, observation, reward, done):
        focus, speedX, speedY, speedZ, opponents, rpm, track, wheelSpinVel = observation

        if np.min(track[:10]) > self.threshold or np.min(track[9:]) > self.threshold:
            
            if np.average(track[:10]) > np.average(track[9:]):
                self.direction = 0.3 * np.exp(-np.average(track[:10]))
            else:
                self.direction = - 0.3 * np.exp(-np.average(track[9:]))
            self.throttle *= 0.8

        else:
            self.direction *= 0.2
            self.throttle *= 1.25

        self.direction = np.clip(self.direction, -1.0, 1.0)
        self.throttle = np.clip(self.throttle, 0.0, 1.0)

        return (self.direction, self.throttle)
