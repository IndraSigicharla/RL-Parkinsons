import numpy as np
import silence_tensorflow.auto
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs.array_spec import ArraySpec, BoundedArraySpec
from tf_agents.trajectories import time_step as ts

class ClassifierEnv(PyEnvironment):
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        self._action_spec = BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=1, name="action")
        self._observation_spec = ArraySpec(shape=X_train.shape[1:], dtype=X_train.dtype, name="observation")
        self._episode_ended = False

        self.X_train = X_train
        self.y_train = y_train
        self.id = np.arange(self.X_train.shape[0])

        self.episode_step = 0
        self._state = self.X_train[self.id[self.episode_step]]

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        np.random.shuffle(self.id) 
        self.episode_step = 0  
        self._state = self.X_train[self.id[self.episode_step]]
        self._episode_ended = False  
        return ts.restart(self._state)

    def _step(self, action: int):
        if self._episode_ended: return self.reset()

        env_action = self.y_train[self.id[self.episode_step]] 
        self.episode_step += 1

        if action == env_action: reward = 1  
        else: 
            reward = -1 
            self._episode_ended = True  

        # print(reward)

        if self.episode_step == self.X_train.shape[0] - 1: self._episode_ended = True
        
        self._state = self.X_train[self.id[self.episode_step]]  

        return ts.termination(self._state, reward) if self._episode_ended else ts.transition(self._state, reward)
