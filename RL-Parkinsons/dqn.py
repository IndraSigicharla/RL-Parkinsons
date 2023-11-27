import pickle
from datetime import datetime
import silence_tensorflow.auto
import numpy as np
import tensorflow as tf
from classifierenv import ClassifierEnv
from metrics import (classification_metrics, decision_function, network_predictions, plot_pr_curve, plot_roc_curve)
from tensorflow import data
from tensorflow.keras.optimizers import Adam
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.sequential import Sequential
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.utils import common
from tqdm import tqdm

class TrainDQN():
    def __init__(self, episodes: int, warmup_steps: int, learning_rate: float, gamma: float, min_epsilon: float, decay_episodes: int,
                 model_path: str = None, log_dir: str = None, batch_size: int = 64, memory_length: int = None,
                 collect_steps_per_episode: int = 1, val_every: int = None, target_update_period: int = 1, target_update_tau: float = 1.0,
                 progressbar: bool = True, n_step_update: int = 1, gradient_clipping: float = 1.0, collect_every: int = 1) -> None:
        self.episodes = episodes  # Total episodes
        self.warmup_steps = warmup_steps  # Amount of warmup steps before training
        self.batch_size = batch_size  # Batch size of Replay Memory
        self.collect_steps_per_episode = collect_steps_per_episode  # Amount of steps to collect data each episode
        self.collect_every = collect_every  # Step interval to collect data during training
        self.learning_rate = learning_rate  # Learning Rate
        self.gamma = gamma  # Discount factor
        self.min_epsilon = min_epsilon  # Minimal chance of choosing random action
        self.decay_episodes = decay_episodes  # Number of episodes to decay from 1.0 to `EPSILON`
        self.target_update_period = target_update_period  # Period for soft updates
        self.target_update_tau = target_update_tau
        self.progressbar = progressbar  # Enable or disable the progressbar for collecting data and training
        self.n_step_update = n_step_update
        self.gradient_clipping = gradient_clipping  # Clip the loss
        self.compiled = False
        NOW = datetime.now().strftime("%Y%m%d_%H%M%S")

        if memory_length is not None:
            self.memory_length = memory_length  # Max Replay Memory length
        else:
            self.memory_length = warmup_steps

        if val_every is not None:
            self.val_every = val_every  # Validate the policy every `val_every` episodes
        else:
            self.val_every = self.episodes // min(50, self.episodes)  # Can't validate the model 50 times if self.episodes < 50

        if model_path is not None:
            self.model_path = model_path
        else:
            self.model_path = "./models/" + "model1" + ".pkl"

        if log_dir is None:
            log_dir = "./logs/" + NOW
        self.writer = tf.summary.create_file_writer(log_dir)

    def compile_model(self, X_train, y_train, layers: list = [], loss_fn=common.element_wise_squared_loss) -> None:

        self.train_env = TFPyEnvironment(ClassifierEnv(X_train, y_train))
        self.global_episode = tf.Variable(0, name="global_episode", dtype=np.int64, trainable=False)  # Global train episode counter

        epsilon_decay = tf.compat.v1.train.polynomial_decay(
            1.0, self.global_episode, self.decay_episodes, end_learning_rate=self.min_epsilon)

        self.q_net = Sequential(layers, self.train_env.observation_spec())

        self.agent = DdqnAgent(self.train_env.time_step_spec(),
                               self.train_env.action_spec(),
                               q_network=self.q_net,
                               optimizer=Adam(learning_rate=self.learning_rate),
                               td_errors_loss_fn=loss_fn,
                               train_step_counter=self.global_episode,
                               target_update_period=self.target_update_period,
                               target_update_tau=self.target_update_tau,
                               gamma=self.gamma,
                               epsilon_greedy=epsilon_decay,
                               n_step_update=self.n_step_update,
                               gradient_clipping=self.gradient_clipping)
                               
        self.agent.initialize()

        self.random_policy = RandomTFPolicy(self.train_env.time_step_spec(), self.train_env.action_spec())
        self.replay_buffer = TFUniformReplayBuffer(data_spec=self.agent.collect_data_spec,
                                                   batch_size=self.train_env.batch_size,
                                                   max_length=self.memory_length)

        self.warmup_driver = DynamicStepDriver(self.train_env,
                                               self.random_policy,
                                               observers=[self.replay_buffer.add_batch],
                                               num_steps=self.warmup_steps)  # Uses a random policy

        self.collect_driver = DynamicStepDriver(self.train_env,
                                                self.agent.collect_policy,
                                                observers=[self.replay_buffer.add_batch],
                                                num_steps=self.collect_steps_per_episode)  # Uses the epsilon-greedy policy of the agent

        self.agent.train = common.function(self.agent.train)  # Optimization
        self.warmup_driver.run = common.function(self.warmup_driver.run)
        self.collect_driver.run = common.function(self.collect_driver.run)

        self.compiled = True

    def train(self, *args) -> None:
        assert self.compiled, "Model must be compiled with model.compile_model(X_train, y_train, layers) before training."

        if self.progressbar:
            print(f"\033[92mCollecting data for {self.warmup_steps:,} steps... This might take a few minutes...\033[0m")

        self.warmup_driver.run(time_step=None, policy_state=self.random_policy.get_initial_state(self.train_env.batch_size))

        if self.progressbar:
            print(f"\033[92m{self.replay_buffer.num_frames():,} frames collected!\033[0m")

        dataset = self.replay_buffer.as_dataset(sample_batch_size=self.batch_size, num_steps=self.n_step_update + 1,
                                                num_parallel_calls=data.experimental.AUTOTUNE).prefetch(data.experimental.AUTOTUNE)
        iterator = iter(dataset)

        def _train():
            experiences, _ = next(iterator)
            return self.agent.train(experiences).loss
        _train = common.function(_train)  # Optimization

        ts = None
        policy_state = self.agent.collect_policy.get_initial_state(self.train_env.batch_size)
        self.collect_metrics(*args)  # Initial collection for step 0
        pbar = tqdm(total=self.episodes, disable=(not self.progressbar), desc="Training the DQN")  # TQDM progressbar
        for _ in range(self.episodes):
            if not self.global_episode % self.collect_every:
                # Collect a few steps using collect_policy and save to `replay_buffer`
                if self.collect_steps_per_episode != 0:
                    ts, policy_state = self.collect_driver.run(time_step=ts, policy_state=policy_state)
                pbar.update(self.collect_every)  # More stable TQDM updates, collecting could take some time

            # Sample a batch of data from `replay_buffer` and update the agent's network
            train_loss = _train()

            if not self.global_episode % self.val_every:
                with self.writer.as_default():
                    tf.summary.scalar("train_loss", train_loss, step=self.global_episode)

                self.collect_metrics(*args)
        pbar.close()

    def collect_metrics(self, X_val: np.ndarray, y_val: np.ndarray, save_best: str = None):
        y_pred = network_predictions(self.agent._target_q_network, X_val)
        stats = classification_metrics(y_val, y_pred)
        avgQ = np.mean(decision_function(self.agent._target_q_network, X_val))  # Max action for each x in X

        if save_best is not None:
            if not hasattr(self, "best_score"):  # If no best model yet
                self.best_score = 0.0

            if stats.get(save_best) >= self.best_score:  # Overwrite best model
                self.save_network()  # Saving directly to avoid shallow copy without trained weights
                self.best_score = stats.get(save_best)

        with self.writer.as_default():
            tf.summary.scalar("AverageQ", avgQ, step=self.global_episode)  # Average Q-value for this epoch
            for k, v in stats.items():
                tf.summary.scalar(k, v, step=self.global_episode)

    def evaluate(self, X_test, y_test, X_train=None, y_train=None):
        if hasattr(self, "best_score"):
            print(f"\033[92mBest score: {self.best_score:6f}!\033[0m")
            network = self.load_network(self.model_path)  # Load best saved model
        else:
            network = self.agent._target_q_network  # Load latest target model

        if (X_train is not None) and (y_train is not None):
            plot_pr_curve(network, X_test, y_test, X_train, y_train)
            plot_roc_curve(network, X_test, y_test, X_train, y_train)

        y_pred = network_predictions(network, X_test)
        return classification_metrics(y_test, y_pred)

    def save_network(self):
        with open(self.model_path, "wb") as f:  # Save Q-network as pickle
            pickle.dump(self.agent._target_q_network, f)

    @staticmethod
    def load_network(fp: str):
        with open(fp, "rb") as f:  # Load the Q-network
            network = pickle.load(f)
        return network
