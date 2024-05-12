from agent import Agent
import numpy as np

class BasicAgent(Agent):
    def __init__(self):
        self.direction = 0
        self.throttle = 1.0
        self.threshold = 0.03875

    def action(self, observation):
        focus, speedX, speedY, speedZ, opponents, rpm, track, wheelSpinVel, laptime, distraced = observation

        if np.min(track) < self.threshold:
            left = np.min(track[:9]) 
            right = np.min(track[10:])
            
            if left > right:
                self.direction = 0.3 * np.exp(-right)
            else:
                self.direction = - 0.3 * np.exp(-left)

            self.throttle *= 0.8
            self.throttle = np.max([0.25, self.throttle])

        else:
            self.direction *= 0.1
            self.throttle *= 1.25

        self.direction = np.clip(self.direction, -1.0, 1.0)
        self.throttle = np.clip(self.throttle, 0.0, 1.0)

        return (self.direction, self.throttle)