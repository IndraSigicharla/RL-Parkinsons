import numpy as np
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
        np.random.shuffle(self.id)  # Shuffle the X and y data
        self.episode_step = 0  # Reset episode step counter at the end of every episode
        self._state = self.X_train[self.id[self.episode_step]]
        self._episode_ended = False  # Reset terminal condition

        return ts.restart(self._state)

    def _step(self, action: int):
        if self._episode_ended:
            return self.reset()

        env_action = self.y_train[self.id[self.episode_step]]  # The label of the current state
        self.episode_step += 1

        if action == env_action:  # Correct action
            reward = 1  # True Positive
        else:  # Incorrect action
            reward = -1  # False Negative
            self._episode_ended = True  # Stop episode when minority class is misclassified

        # print(reward)

        if self.episode_step == self.X_train.shape[0] - 1:  # If last step in data
            self._episode_ended = True

        self._state = self.X_train[self.id[self.episode_step]]  # Update state with new datapoint

        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward)
